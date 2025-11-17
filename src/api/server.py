"""
Grant Discovery API Server

RESTful API for searching and exploring grant opportunities.

Endpoints:
    GET  /health              - Health check
    GET  /grants              - List grants
    GET  /grants/{grant_id}   - Get grant details
    GET  /search              - Semantic search
    POST /search/explain      - LLM-powered search explanation
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    HealthResponse,
    GrantSummary,
    GrantDetail,
    GrantWithDocuments,
    DocumentSummary,
    SearchResponse,
    SearchHit,
    ExplainRequest,
    ExplainResponse,
    ReferencedGrant,
    ChatTurn,
    ChatRequest,
    ChatGrant,
    ChatResponse,
)
from src.storage.grant_store import GrantStore
from src.storage.document_store import DocumentStore
from src.storage.explanation_cache import ExplanationCache
from src.index.vector_index import VectorIndex
from src.core.domain_models import Grant


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Grant Discovery API",
    version="1.0.0",
    description="Search and explore grant opportunities. Powered by GPT-5 for intelligent explanations.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize storage and index
DB_PATH = "grants.db"
grant_store = GrantStore(DB_PATH)
doc_store = DocumentStore(DB_PATH)
vector_index = VectorIndex(db_path=DB_PATH)
explanation_cache = ExplanationCache(DB_PATH)

# LLM client (initialized on first use)
llm_client: Optional['LLMClient'] = None

# Chat LLM client (initialized lazily)
chat_llm_client = None


# -----------------------------------------------------------------------------
# Expert Examples Functions
# -----------------------------------------------------------------------------

def get_expert_examples(category: str = None, limit: int = 3, min_quality: int = 4):
    """
    Retrieve expert examples from database.

    Args:
        category: Filter by category (optional)
        limit: Max number of examples to return
        min_quality: Minimum quality score (1-5)

    Returns:
        List of expert example dicts
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    if category:
        cur.execute("""
            SELECT user_query, expert_response, category, grant_mentioned, client_context
            FROM expert_examples
            WHERE category = ? AND is_active = 1 AND quality_score >= ?
            ORDER BY quality_score DESC, added_date DESC
            LIMIT ?
        """, (category, min_quality, limit))
    else:
        cur.execute("""
            SELECT user_query, expert_response, category, grant_mentioned, client_context
            FROM expert_examples
            WHERE is_active = 1 AND quality_score >= ?
            ORDER BY quality_score DESC, added_date DESC
            LIMIT ?
        """, (min_quality, limit))

    examples = []
    for row in cur.fetchall():
        examples.append({
            "user_query": row[0],
            "expert_response": row[1],
            "category": row[2],
            "grant_mentioned": row[3],
            "client_context": row[4]
        })

    conn.close()
    return examples


def format_expert_examples_for_prompt(examples: list) -> str:
    """Format expert examples for inclusion in system prompt."""
    if not examples:
        return ""

    formatted = "\n---EXPERT RESPONSE EXAMPLES---\n\n"
    formatted += "Learn from these examples of how to respond:\n\n"

    for i, ex in enumerate(examples, 1):
        formatted += f"EXAMPLE {i}"
        if ex['category']:
            formatted += f" ({ex['category']})"
        if ex['grant_mentioned']:
            formatted += f" - {ex['grant_mentioned']}"
        formatted += "\n\n"

        if ex['client_context']:
            formatted += f"Client context: {ex['client_context']}\n\n"

        formatted += f"User Query: {ex['user_query']}\n\n"
        formatted += f"Expert Response:\n{ex['expert_response'][:800]}...\n\n"
        formatted += "---\n\n"

    return formatted


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def _get_grant_summary(grant_id: str) -> Optional[str]:
    """
    Fetch cached GPT summary for a grant.

    Args:
        grant_id: Grant ID to look up

    Returns:
        Summary text if available, None otherwise
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT summary FROM grant_summaries WHERE grant_id = ?",
            (grant_id,)
        )
        row = cur.fetchone()
        conn.close()

        if row:
            return row[0]
    except Exception as e:
        # Don't break search if summary lookup fails
        logger.error(f"Failed to load summary for {grant_id}: {e}")

    return None


def _build_snippet(text: str, max_len: int = 300) -> str:
    """
    Build snippet from document text (fallback when no summary).

    Args:
        text: Document text
        max_len: Maximum snippet length

    Returns:
        Truncated snippet with ellipsis if needed
    """
    text = text.strip()
    if len(text) <= max_len:
        return text

    # Truncate at sentence boundary if possible
    truncated = text[:max_len]
    last_period = truncated.rfind('.')
    if last_period > max_len * 0.5:  # Only if we don't lose too much
        return text[:last_period + 1]

    return truncated.rstrip() + "..."


def _grant_to_summary(grant: Grant) -> GrantSummary:
    """Convert Grant domain object to GrantSummary schema."""
    return GrantSummary(
        id=grant.id,
        title=grant.title,
        url=grant.url,
        source=grant.source,
        total_fund=grant.total_fund,
        closes_at=grant.closes_at.isoformat() if grant.closes_at else None,
        is_active=grant.is_active,
        tags=grant.tags or [],
    )


# Scoring thresholds for grant filtering
MIN_SCORE_STRONG = 0.55
MIN_SCORE_WEAK = 0.48
MAX_GRANTS = 3  # Show up to 3 relevant grants (quality over quantity)

# System prompt for neutral search assistant
SYSTEM_PROMPT = """You are Ailsa, a UK research funding advisor specializing in NIHR and Innovate UK grants.

RESPONSE QUANTITY & LENGTH:
- Show 2-3 grants max in your initial response, not all available options
- Pick the MOST relevant grants based on the user's query
- Users can always ask "what else?" or "show me more" if they want additional options
- Quality over quantity - better to explain 2-3 grants well than 5+ poorly
- **BE CONCISE**: Aim for 300-500 words maximum
- Cut filler words and repetition

RESPONSE STYLE FLEXIBILITY:
- The examples show TONE and PATTERNS, not a rigid template
- DON'T force every response into "Ailsa's Take / Why it's a fit / Next steps" structure
- Adapt your format based on the question:
  * Simple query → Simple answer (2-3 paragraphs)
  * Complex query → Structured breakdown
  * Follow-up → Direct answer without repeating structure
- Use natural conversation flow

WHEN TO USE STRUCTURED FORMAT:
✅ User asks: "Show me grants for X" → Use structure for 2-3 grants
✅ User asks: "What's the best grant for Y?" → Use structure for top pick
❌ User asks: "What's the deadline?" → Just answer directly, no structure
❌ User asks: "How do I apply?" → Explain process, no "Ailsa's Take" needed

EXAMPLE FLEXIBLE RESPONSES:

Query: "What's the deadline for Biomedical Catalyst?"
Response: "The current Biomedical Catalyst round closes December 10, 2025 - that's 22 days from now. Given the competitive nature and detailed application requirements, I'd recommend starting your proposal this week if you're serious about applying."

Query: "Should I apply for loans or grants?"
Response: "Depends on your financial position and timeline. Grants are non-repayable but highly competitive (10-20% success rates). Innovation Loans give you more flexibility with 3.7% rates during the project, but Innovate UK's Credit team will scrutinize your ability to repay. What's your current runway and revenue situation?"

Query: "Show me medical device grants"
Response: [Use full structured format for 2-3 top grants]

FORMATTING RULES:
- Use ## for grant section headers (when appropriate)
- Use **bold** for emphasis
- Keep paragraphs to 2-3 sentences
- Use bullet lists for details when helpful
- NO repetition of funding/deadline info (shown in grant cards below)

CRITICAL:
- The grant cards below your response show funding amounts and deadlines
- Your text should add insight and strategy, NOT repeat metadata
- Focus on eligibility, application tips, and strategic advice
- If grants aren't perfect matches, say so clearly

TONE & STYLE:
- Professional but warm
- Direct and actionable
- Use "you" to speak to the user
- Vary your style based on the question type

HANDLING EDGE CASES:
- If NO grants match but question is about general funding: Provide brief helpful guidance
- If user asks about specific grant by name: Find it and discuss in detail
- If comparing grants: Highlight key differences clearly

You MUST respond in valid JSON format with exactly these keys:
{
  "answer_markdown": "Your concise markdown response (300-500 words max)",
  "recommended_grants": [
    {
      "grant_id": "grant_id_here",
      "title": "grant title",
      "source": "innovate_uk or nihr",
      "reason": "brief explanation of relevance (1-2 sentences)"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """User question:
{query}

Available grants from semantic search (ranked by relevance):
{grant_summaries}

Your task:
1. Write a COMPREHENSIVE response that addresses the user's question thoroughly
2. Analyze ALL grants provided and discuss the most relevant ones in detail
3. Include specific information: funding amounts, deadlines, eligibility, and why each grant matches
4. Select up to 5 most relevant grants for the recommended_grants list (score >= {min_score_strong})
5. Do NOT hallucinate grants - only reference grants provided above

IMPORTANT INSTRUCTIONS:
- Use the grant information above to provide detailed, actionable advice
- Explain WHY each grant is relevant to the user's specific question
- If grants are closed, mention them and explain when similar opportunities might be available
- Include funding amounts and deadlines prominently
- Use markdown formatting for readability (bold for key info, bullets for lists)
- Be thorough but well-organized - aim for 3-8 paragraphs depending on complexity
- The grant cards will appear below your response, so provide context and analysis, not just repetition

Respond in valid JSON with keys:
- "answer_markdown": comprehensive markdown-formatted response that thoroughly addresses the query
- "recommended_grants": a list (max 5) of the most relevant grants with:
    - grant_id
    - title
    - source
    - reason (2-3 sentences explaining relevance and fit)
"""


def apply_semantic_boost(query: str, title: str, base_score: float) -> float:
    """
    Apply semantic boosting based on query and grant title keywords.

    Args:
        query: User's search query
        title: Grant title
        base_score: Base relevance score

    Returns:
        Boosted score
    """
    q = query.lower()
    t = title.lower()

    score = base_score

    # Cancer / oncology boosting
    if "cancer" in q or "oncolog" in q:
        if "cancer" in t or "oncolog" in t:
            score *= 1.12

    # Therapeutics / therapy
    if "therap" in q:
        if "therap" in t or "treatment" in t:
            score *= 1.08

    # Paediatrics / pediatrics
    if "paediatr" in q or "pediatr" in q:
        if "paediatr" in t or "pediatr" in t or "children" in t or "child" in t:
            score *= 1.10

    # AI / agentic / LLM
    if any(term in q for term in ["ai", "artificial intelligence", "llm", "agentic"]):
        if any(term in t for term in ["ai", "artificial intelligence", "agentic", "llm"]):
            score *= 1.10

    return score


def select_top_grants(hits, query: str = ""):
    """
    Filter and deduplicate grants by score threshold with semantic boosting.

    Args:
        hits: List of search results with grant_id, score, metadata
        query: User query for semantic boosting

    Returns:
        List of top grant summaries (up to MAX_GRANTS, or more if user requests)
    """
    from collections import defaultdict

    by_grant = defaultdict(list)

    for h in hits:
        if not h.grant_id:
            continue
        by_grant[h.grant_id].append(h)

    items = []
    for gid, group in by_grant.items():
        best = max(group, key=lambda x: x.score)

        # Get grant details
        grant = grant_store.get_grant(gid)
        if not grant:
            continue

        # Filter out Smart Grants (paused January 2025)
        title_lower = grant.title.lower()
        if "smart grant" in title_lower or "smart grants" in title_lower:
            logger.info(f"Filtered out Smart Grant: {grant.title}")
            continue

        # Apply semantic boosting
        boosted_score = apply_semantic_boost(query, grant.title, float(best.score))

        items.append({
            "grant_id": gid,
            "title": grant.title,
            "source": grant.source,
            "status": "open" if grant.is_active else "closed",
            "closes_at": grant.closes_at.isoformat() if grant.closes_at else None,
            "total_fund_gbp": getattr(grant, "total_fund_gbp", None) or grant.total_fund,
            "best_score": boosted_score,
            "url": grant.url,
        })

    # Separate open and closed grants
    from datetime import datetime, timezone

    open_grants = []
    closed_grants = []

    for item in items:
        if item["best_score"] < MIN_SCORE_STRONG:
            continue  # Skip low-relevance grants entirely

        # Check if grant is truly open based on deadline
        is_open = item["status"] == "open"
        if item["closes_at"]:
            try:
                deadline_dt = datetime.fromisoformat(item["closes_at"].replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                is_open = deadline_dt > now
            except:
                pass  # Keep original status if parsing fails

        if is_open:
            open_grants.append(item)
        else:
            closed_grants.append(item)

    # Sort each group by score
    open_grants.sort(key=lambda x: x["best_score"], reverse=True)
    closed_grants.sort(key=lambda x: x["best_score"], reverse=True)

    # Check if user wants comprehensive list
    query_lower = query.lower()
    max_grants = MAX_GRANTS
    if any(phrase in query_lower for phrase in ['all grants', 'show me everything', 'complete list', 'all options', 'show more', 'what else']):
        max_grants = 5
        logger.info(f"User requested comprehensive list, showing up to {max_grants} grants")

    # Prioritize open grants - only show closed if very few open options
    relevant = open_grants[:max_grants]

    # Only add closed grants if we have 1 or fewer open grants (desperate measure)
    if len(relevant) <= 1 and closed_grants:
        # Add up to 1 closed grant as context only
        relevant.extend(closed_grants[:1])
        logger.info(f"⚠️ Added {len(closed_grants[:1])} closed grant(s) due to limited open options")

    logger.info(f"Selected {len(open_grants)} open and {len(closed_grants)} closed grants, returning {len(relevant)} total (open only: {len([g for g in relevant if g in open_grants])})")

    return relevant


def build_llm_context(query: str, hits, grants_for_llm):
    """
    Build comprehensive LLM context from selected grants with rich detail.

    Args:
        query: User query
        hits: All search hits
        grants_for_llm: Selected grant summaries

    Returns:
        Formatted context string with detailed grant information
    """
    from datetime import datetime, timezone

    selected_ids = {g["grant_id"] for g in grants_for_llm}

    # Group hits by grant to combine information
    from collections import defaultdict
    grant_hits = defaultdict(list)

    for h in hits:
        if h.grant_id in selected_ids:
            grant_hits[h.grant_id].append(h)

    context_blocks = []

    for grant_summary in grants_for_llm:
        gid = grant_summary["grant_id"]
        grant = grant_store.get_grant(gid)

        if not grant:
            continue

        # Calculate deadline urgency
        deadline_display = "Not specified"
        urgency_note = ""
        if grant.closes_at:
            deadline_display = grant.closes_at.isoformat()
            try:
                now = datetime.now(timezone.utc)
                deadline_dt = grant.closes_at
                if deadline_dt.tzinfo is None:
                    from datetime import timezone
                    deadline_dt = deadline_dt.replace(tzinfo=timezone.utc)

                days_until = (deadline_dt - now).days

                if days_until < 0:
                    urgency_note = " (⚠️ CLOSED)"
                elif days_until < 30:
                    urgency_note = f" (⚠️ Closing soon: {days_until} days remaining)"
                elif days_until < 90:
                    urgency_note = f" ({days_until} days remaining)"
            except:
                pass

        # Format funding amount
        funding_amount = getattr(grant, 'total_fund_gbp', None) or grant.total_fund
        if funding_amount:
            if isinstance(funding_amount, (int, float)):
                funding_display = f"£{funding_amount:,.0f}"
            else:
                funding_display = str(funding_amount)
        else:
            funding_display = "Not specified"

        # Get best snippet from hits
        relevant_snippets = []
        for h in grant_hits[gid]:
            if hasattr(h, 'text') and h.text:
                doc_type = h.metadata.get("doc_type", "general") if hasattr(h, 'metadata') else "general"
                snippet_text = h.text[:600].strip()
                if snippet_text:
                    relevant_snippets.append(f"  [{doc_type.upper()}] {snippet_text}")

        snippets_text = "\n".join(relevant_snippets[:3]) if relevant_snippets else "  No additional context available"

        # Build rich context block
        context_blocks.append(f"""
═══════════════════════════════════════════════════════════
GRANT | Relevance Score: {grant_summary['best_score']:.3f}
═══════════════════════════════════════════════════════════

Title: {grant.title}
Grant ID: {gid}
Source: {grant.source.upper()}
Status: {grant_summary['status'].upper()}{urgency_note}

💰 Funding Amount: {funding_display}
📅 Deadline: {deadline_display}{urgency_note}
🔗 URL: {grant.url or 'Not available'}

Relevant Context Snippets:
{snippets_text}

---
""")

    header = f"""Found {len(grants_for_llm)} highly relevant grants ranked by semantic similarity.
Use ALL of this information to provide a comprehensive, thorough response.
Higher relevance scores indicate better matches to the user's query.

IMPORTANT: Analyze all grants below and provide detailed insights about each relevant one.

"""

    return header + "\n".join(context_blocks)


def build_user_prompt(query, grants_for_llm):
    """Build user prompt with grant summaries."""
    if grants_for_llm:
        lines = []
        for g in grants_for_llm:
            lines.append(
                f"- {g['title']} (id={g['grant_id']}, "
                f"source={g['source']}, score={g['best_score']:.3f}, "
                f"status={g['status']}, closes_at={g['closes_at']}, "
                f"funding={g['total_fund_gbp']})"
            )
        summary = "\n".join(lines)
    else:
        summary = "(no suitable grants found above the score thresholds)"

    return USER_PROMPT_TEMPLATE.format(
        query=query,
        grant_summaries=summary,
        min_score_strong=MIN_SCORE_STRONG,
    )


def explain_with_gpt(client, query: str, hits):
    """
    Generate explanation using GPT with filtered grants.

    Args:
        client: LLM client
        query: User query
        hits: Search results

    Returns:
        Tuple of (answer_markdown, recommended_grants)
    """
    import json

    # Step 1: Select top grants with semantic boosting
    grants = select_top_grants(hits, query=query)

    # Step 2: Build context
    context = build_llm_context(query, hits, grants)

    # Step 3: Call GPT
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                build_user_prompt(query, grants)
                + "\n\nContext from grant documents:\n"
                + context
            ),
        },
    ]

    raw = client.chat(
        messages=messages,
        temperature=0.3,  # Lower for more focused and consistent responses
        max_tokens=800,  # Reduced for conciseness
    )

    # Parse JSON response
    try:
        data = json.loads(raw)
    except Exception:
        # Fallback: wrap plain text
        data = {
            "answer_markdown": raw,
            "recommended_grants": [
                {
                    "grant_id": g["grant_id"],
                    "title": g["title"],
                    "source": g["source"],
                    "reason": f"Relevance score: {g['best_score']:.3f}"
                }
                for g in grants[:3]
            ],
        }

    answer_markdown = data.get("answer_markdown", "").strip()
    recs = data.get("recommended_grants", []) or []

    # Enrich recommendations with scores from the original grants list
    grant_scores = {g["grant_id"]: g["best_score"] for g in grants}
    for rec in recs:
        if "grant_id" in rec and rec["grant_id"] in grant_scores:
            rec["best_score"] = grant_scores[rec["grant_id"]]
            rec["url"] = next((g["url"] for g in grants if g["grant_id"] == rec["grant_id"]), "#")

    return answer_markdown, recs


def _chat_retrieve(
    query_text: str,
    top_k: int,
    active_only: bool,
    sources: Optional[List[str]],
) -> tuple:
    """
    Retrieve relevant grants for chat using vector search.

    Args:
        query_text: User's question/query
        top_k: Number of results to retrieve
        active_only: Filter to active grants only
        sources: Optional source filter (innovate_uk, nihr)

    Returns:
        Tuple of (hits, grants_by_id)
    """
    # Use existing vector index
    hits = vector_index.query(
        query_text=query_text,
        top_k=top_k * 2,  # Over-fetch for filtering
        filter_scope=None
    )

    # Load and filter grants
    grants_by_id = {}

    for hit in hits:
        gid = hit.grant_id
        if not gid or gid in grants_by_id:
            continue

        grant = grant_store.get_grant(gid)
        if not grant:
            continue

        # Apply filters
        if active_only and not grant.is_active:
            continue

        if sources and grant.source not in sources:
            continue

        grants_by_id[gid] = grant

        # Stop when we have enough
        if len(grants_by_id) >= top_k:
            break

    # Filter hits to only those with valid grants
    filtered_hits = [h for h in hits if h.grant_id in grants_by_id]

    return filtered_hits, grants_by_id


def group_results_by_grant(hits: list, max_grants: int = 5) -> list:
    """
    Group document-level search hits by grant and aggregate content.

    Args:
        hits: List of document-level search results
        max_grants: Maximum number of grants to return

    Returns:
        List of grant groups with aggregated content
    """
    from collections import defaultdict
    from datetime import datetime

    # Group hits by grant_id
    grant_groups = defaultdict(list)
    for hit in hits:
        if hit.grant_id:
            grant_groups[hit.grant_id].append(hit)

    # Build aggregated grant objects
    results = []

    for grant_id, grant_hits in grant_groups.items():
        # Get grant metadata
        grant = grant_store.get_grant(grant_id)
        if not grant:
            continue

        # Calculate max score for this grant
        max_score = max(h.score for h in grant_hits)

        # Aggregate text from all documents (limit to 4000 chars)
        texts = []
        total_chars = 0
        for hit in sorted(grant_hits, key=lambda h: h.score, reverse=True):
            if total_chars >= 4000:
                break
            chunk = hit.text[:1000]  # Limit each chunk
            texts.append(chunk)
            total_chars += len(chunk)

        aggregated_text = "\n\n".join(texts)

        # Get or generate description
        description = _get_grant_summary(grant_id) or grant.description or ""

        # Determine status
        status = "open" if grant.is_active else "closed"
        if grant.opens_at and grant.opens_at > datetime.now():
            status = "upcoming"

        results.append({
            "grant_id": grant_id,
            "score": max_score,
            "title": grant.title,
            "source": grant.source,
            "funding": grant.total_fund or "Not specified",
            "status": status,
            "is_active": grant.is_active,
            "deadline": grant.closes_at.isoformat() if grant.closes_at else None,
            "opens_at": grant.opens_at.isoformat() if grant.opens_at else None,
            "closes_at": grant.closes_at.isoformat() if grant.closes_at else None,
            "description": description,
            "aggregated_text": aggregated_text,
            "documents": grant_hits,
            "url": grant.url
        })

    # Sort by score and take top N
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_grants]


def _grant_to_detail(grant: Grant) -> GrantDetail:
    """Convert Grant domain object to GrantDetail schema."""
    return GrantDetail(
        id=grant.id,
        title=grant.title,
        url=grant.url,
        source=grant.source,
        description=grant.description,
        total_fund=grant.total_fund,
        project_size=grant.project_size,
        opens_at=grant.opens_at.isoformat() if grant.opens_at else None,
        closes_at=grant.closes_at.isoformat() if grant.closes_at else None,
        is_active=grant.is_active,
        funding_rules=grant.funding_rules or {},
        tags=grant.tags or [],
    )


def _truncate(text: str, max_chars: int = 900) -> str:
    """Truncate text to max_chars, adding ellipsis if needed."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


# -----------------------------------------------------------------------------
# Website Parsing Functions
# -----------------------------------------------------------------------------

def extract_url_from_message(message: str) -> Optional[str]:
    """
    Extract URL from user message.

    Args:
        message: User message text

    Returns:
        Extracted URL or None if not found
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    match = re.search(url_pattern, message)
    return match.group(0) if match else None


def parse_company_website(url: str) -> Optional[dict]:
    """
    Parse company website to extract context for grant matching.

    Args:
        url: Company website URL

    Returns:
        Dict with extracted context or None if parsing fails
    """
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; AilsaBot/1.0)'},
            allow_redirects=True
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '') if meta_desc else ''

        # Extract title
        title = soup.find('title')
        title_text = title.text.strip() if title else ''

        # Extract main body text (first 2000 chars for better context)
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        body_text = soup.get_text(separator=' ', strip=True)
        body_text = ' '.join(body_text.split())[:2000]

        # Combine for context
        full_text = f"{title_text} {description} {body_text}"

        # Extract keywords
        keywords = extract_keywords(full_text)

        # Try to detect company stage
        stage = detect_company_stage(full_text)

        # Detect sector/domain
        sector = detect_sector_from_text(full_text)

        # Detect TRL indicators
        trl_estimate = estimate_trl_from_text(full_text)

        # Suggest likely grant matches based on content
        suggested_grants = suggest_grants_from_context(full_text, keywords, sector)

        logger.info(f"✓ Parsed {url}: sector={sector}, stage={stage}, TRL~{trl_estimate}, grants={suggested_grants}")

        return {
            'url': url,
            'title': title_text,
            'description': description,
            'keywords': keywords,
            'stage': stage,
            'sector': sector,
            'trl_estimate': trl_estimate,
            'suggested_grants': suggested_grants,
            'full_context': full_text[:500]
        }

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch website {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to parse website {url}: {e}")
        return None


def extract_keywords(text: str) -> list:
    """
    Extract relevant technology and industry keywords from text.

    Args:
        text: Text to extract keywords from

    Returns:
        List of found keywords
    """
    # Technology and industry keywords relevant for UK grants
    tech_keywords = [
        'AI', 'artificial intelligence', 'machine learning', 'deep learning',
        'healthcare', 'medical device', 'biotech', 'biotechnology',
        'pharma', 'pharmaceutical', 'digital health', 'healthtech',
        'clinical', 'diagnostics', 'therapeutics', 'drug discovery',
        'SaaS', 'platform', 'software', 'data science',
        'regenerative medicine', 'gene therapy', 'immunotherapy',
        'precision medicine', 'personalized medicine',
        'medical imaging', 'wearable', 'IoT', 'sensors',
        'robotics', 'automation', 'manufacturing',
        'clean tech', 'renewable energy', 'sustainability',
        'fintech', 'edtech', 'agritech'
    ]

    text_lower = text.lower()
    found = [kw for kw in tech_keywords if kw.lower() in text_lower]

    return list(set(found))[:10]  # Return up to 10 unique keywords


def detect_sector_from_text(text: str) -> str:
    """
    Detect primary sector from website text.

    Args:
        text: Website text

    Returns:
        Detected sector or 'general'
    """
    text_lower = text.lower()

    # Health/medical sectors
    if any(term in text_lower for term in ['medical device', 'medtech', 'healthcare', 'clinical', 'diagnostic', 'therapeutics']):
        return 'medtech'
    elif any(term in text_lower for term in ['biotech', 'drug discovery', 'pharmaceutical', 'gene therapy']):
        return 'biotech'
    elif any(term in text_lower for term in ['digital health', 'healthtech', 'telemedicine', 'remote monitoring']):
        return 'digital health'

    # Tech sectors
    elif any(term in text_lower for term in ['artificial intelligence', 'machine learning', 'ai platform', 'llm']):
        return 'ai'
    elif any(term in text_lower for term in ['software', 'saas', 'platform', 'api']):
        return 'software'

    # Engineering/materials
    elif any(term in text_lower for term in ['materials science', 'advanced materials', 'composites']):
        return 'materials'
    elif any(term in text_lower for term in ['manufacturing', 'automation', 'robotics']):
        return 'manufacturing'

    # Other sectors
    elif any(term in text_lower for term in ['clean tech', 'renewable', 'sustainability', 'carbon']):
        return 'cleantech'
    elif any(term in text_lower for term in ['fintech', 'financial technology', 'payments']):
        return 'fintech'

    return 'general'


def estimate_trl_from_text(text: str) -> str:
    """
    Estimate Technology Readiness Level from website text.

    Args:
        text: Website text

    Returns:
        TRL estimate or 'unknown'
    """
    text_lower = text.lower()

    # High TRL (7-9) - Commercial/market indicators
    if any(term in text_lower for term in ['available now', 'buy now', 'customers', 'in production', 'commercially available']):
        return '7-9'

    # Mid TRL (4-6) - Prototype/validation indicators
    elif any(term in text_lower for term in ['pilot', 'prototype', 'clinical trial', 'validation', 'demonstrator']):
        return '4-6'

    # Low TRL (1-3) - Research/concept indicators
    elif any(term in text_lower for term in ['research', 'concept', 'feasibility', 'proof of concept', 'early stage']):
        return '1-3'

    return 'unknown'


def suggest_grants_from_context(text: str, keywords: list, sector: str) -> list:
    """
    Suggest likely grant matches based on company context.

    Args:
        text: Full website text
        keywords: Extracted keywords
        sector: Detected sector

    Returns:
        List of suggested grant types
    """
    text_lower = text.lower()
    suggestions = []

    # NIHR grants
    if sector in ['medtech', 'biotech', 'digital health']:
        if 'clinical' in text_lower or 'nhs' in text_lower or 'patient' in text_lower:
            suggestions.append('NIHR i4i')
        if 'research' in text_lower:
            suggestions.append('NIHR Research')

    # Innovate UK grants
    if sector in ['ai', 'software', 'manufacturing', 'materials', 'cleantech']:
        suggestions.append('Innovate UK sector competitions')

    # Biomedical Catalyst (cross-over)
    if sector in ['medtech', 'biotech', 'digital health'] and any(term in text_lower for term in ['commercial', 'market', 'product']):
        suggestions.append('Biomedical Catalyst')

    # KTP if mentions partnerships/collaboration
    if any(term in text_lower for term in ['university', 'research partner', 'collaboration', 'academic']):
        suggestions.append('Knowledge Transfer Partnership')

    return suggestions[:3]  # Top 3 suggestions


def detect_company_stage(text: str) -> str:
    """
    Detect company stage from website text.

    Args:
        text: Website text

    Returns:
        Detected stage: 'early-stage', 'growth-stage', 'established', or 'unknown'
    """
    text_lower = text.lower()

    # Early-stage indicators
    early_indicators = ['seed', 'pre-seed', 'startup', 'launching', 'founded in 202']
    if any(term in text_lower for term in early_indicators):
        return 'early-stage'

    # Growth-stage indicators
    growth_indicators = ['series a', 'series b', 'series c', 'scale', 'scaling', 'growth']
    if any(term in text_lower for term in growth_indicators):
        return 'growth-stage'

    # Established indicators
    established_indicators = ['established', 'leading', 'years of experience', 'decades', 'industry leader']
    if any(term in text_lower for term in established_indicators):
        return 'established'

    return 'unknown'


def analyze_query_intent(query: str) -> str:
    """
    Figure out what stage/intent the user has.

    Args:
        query: User query text

    Returns:
        Intent category string
    """
    query_lower = query.lower()

    # Early exploration
    if any(word in query_lower for word in ['what', 'which', 'options', 'available']):
        return "exploring_options"

    # Specific grant questions
    if any(word in query_lower for word in ['deadline', 'how much', 'criteria', 'eligible']):
        return "specific_requirements"

    # Strategy questions
    if any(word in query_lower for word in ['how to', 'tips', 'advice', 'strategy']):
        return "seeking_strategy"

    # Technical/stage questions
    if 'trl' in query_lower or 'stage' in query_lower:
        return "technical_readiness"

    # Comparison
    if any(word in query_lower for word in ['vs', 'versus', 'or', 'better']):
        return "comparing_options"

    return "general_inquiry"


def adjust_temperature_for_conversation(conversation_length: int, user_message_length: int) -> float:
    """
    Adjust temperature based on conversation dynamics to match user's communication style.

    Args:
        conversation_length: Number of messages in conversation
        user_message_length: Length of current user message

    Returns:
        Appropriate temperature value
    """
    # If user writes short messages, be more focused
    if user_message_length < 50:
        return 0.5  # More focused, less variation

    # If deep in conversation, be more brief and precise
    if conversation_length > 10:
        return 0.4  # Avoid repetition, stay focused

    # First few messages can be warmer and more exploratory
    if conversation_length < 3:
        return 0.7  # More natural variation

    return 0.6  # Default balanced


def format_known_facts(facts: dict) -> str:
    """
    Format known facts into a readable string for the prompt.

    Args:
        facts: Dictionary of known facts

    Returns:
        Formatted string of known facts
    """
    known = []
    if facts.get('trl'):
        known.append(f"TRL: {facts['trl']}")
    if facts.get('budget'):
        known.append(f"Budget: {facts['budget']}")
    if facts.get('sector'):
        known.append(f"Sector: {facts['sector']}")
    if facts.get('company_type'):
        known.append(f"Company type: {facts['company_type']}")
    if facts.get('stage'):
        known.append(f"Stage: {facts['stage']}")
    if facts.get('timeline'):
        known.append(f"Timeline: {facts['timeline']}")
    if facts.get('clinical_champion'):
        known.append("Has clinical champion")

    if not known:
        return "None yet - this is a fresh conversation"

    return ", ".join(known)


class GrantMentionDetector:
    """Detect and track grant mentions in real-time responses."""

    def __init__(self, grant_store):
        self.grant_store = grant_store

        # Common grant name patterns and aliases
        self.grant_aliases = {
            'i4i': ['i4i', 'invention for innovation', 'i4i pda', 'i4i product development'],
            'biomedical_catalyst': ['biomedical catalyst', 'biomed catalyst', 'bmc'],
            'smart_grants': ['smart grants', 'smart grant'],
            'innovation_loans': ['innovation loan', 'innovate uk loan'],
            'ktp': ['ktp', 'knowledge transfer partnership'],
            'sbri': ['sbri', 'small business research initiative'],
        }

    def extract_grant_mentions(self, text: str, available_grants: list = None) -> list:
        """
        Find all grant mentions in text and match to actual grants.

        Args:
            text: Response text to analyze
            available_grants: List of grants from search results

        Returns:
            List of matched grant objects
        """
        mentioned_grants = []
        text_lower = text.lower()

        # First, check available grants from search results
        if available_grants:
            for grant in available_grants:
                grant_title = grant.get('title', '').lower()
                # Check if grant title is mentioned in response
                if len(grant_title) > 10 and grant_title in text_lower:
                    if grant not in mentioned_grants:
                        mentioned_grants.append(grant)

        # Then check for common aliases
        for grant_key, aliases in self.grant_aliases.items():
            for alias in aliases:
                if alias.lower() in text_lower:
                    # Try to find matching grant from available_grants
                    if available_grants:
                        for grant in available_grants:
                            grant_title_lower = grant.get('title', '').lower()
                            # Match alias to grant title
                            if grant_key in grant_title_lower or any(a in grant_title_lower for a in aliases):
                                if grant not in mentioned_grants:
                                    mentioned_grants.append(grant)
                                    break

        return mentioned_grants[:5]  # Max 5 grant cards


def validate_and_correct_trl(query: str) -> dict:
    """
    Validate TRL mentioned in query and provide correction if invalid.

    Args:
        query: User query text

    Returns:
        Dict with validation result and optional correction
    """
    trl_match = re.search(r'trl\s*(\d+)', query.lower())
    if trl_match:
        try:
            trl = int(trl_match.group(1))

            if trl > 9:
                return {
                    'valid': False,
                    'trl': None,
                    'correction': f"Quick note - TRL only goes up to 9 (market deployment). "
                                 f"If you meant TRL {min(trl % 10, 9)}, that's already commercial stage. "
                                 f"What's your actual development status?"
                }
            elif trl < 1:
                return {
                    'valid': False,
                    'trl': None,
                    'correction': "TRL starts at 1 (basic research). Where are you really at?"
                }
            else:
                return {'valid': True, 'trl': trl, 'correction': None}
        except ValueError:
            pass

    return {'valid': True, 'trl': None, 'correction': None}


def extract_conversation_facts(history: list) -> dict:
    """
    Track key facts already shared in conversation to avoid asking twice.

    Args:
        history: List of conversation messages

    Returns:
        Dictionary of known facts
    """
    facts = {
        'trl': None,
        'budget': None,
        'company_type': None,
        'clinical_champion': None,
        'timeline': None,
        'partnerships': [],
        'sector': None,
        'stage': None
    }

    if not history:
        return facts

    for turn in history:
        content = turn.content.lower()

        # Track TRL (with validation)
        if 'trl' in content:
            trl_match = re.search(r'trl\s*(\d+)', content)
            if trl_match:
                try:
                    trl = int(trl_match.group(1))
                    # Only store valid TRL values (1-9)
                    if 1 <= trl <= 9:
                        facts['trl'] = trl
                except ValueError:
                    pass

        # Track budget
        if '£' in content:
            budget_match = re.search(r'£([\d,]+)([km])?', content)
            if budget_match:
                facts['budget'] = budget_match.group(0)

        # Track company type
        if 'sme' in content or 'small medium enterprise' in content:
            facts['company_type'] = 'SME'
        elif 'startup' in content or 'start-up' in content:
            facts['company_type'] = 'startup'

        # Track clinical champion
        if 'clinical champion' in content or 'clinical lead' in content:
            facts['clinical_champion'] = True

        # Track timeline/urgency
        if 'urgent' in content or 'asap' in content or 'quickly' in content:
            facts['timeline'] = 'urgent'
        elif 'months' in content:
            months_match = re.search(r'(\d+)\s*months?', content)
            if months_match:
                facts['timeline'] = f"{months_match.group(1)} months"

        # Track sector
        sectors = ['materials', 'biomedical', 'medtech', 'ai', 'software', 'healthtech', 'diagnostics']
        for sector in sectors:
            if sector in content:
                facts['sector'] = sector
                break

        # Track stage
        if 'pre-seed' in content or 'preseed' in content:
            facts['stage'] = 'pre-seed'
        elif 'seed' in content:
            facts['stage'] = 'seed'
        elif 'series a' in content:
            facts['stage'] = 'series-a'

    return facts


def get_smart_followup(query: str, grants: list, conversation_history: list, known_facts: dict = None) -> Optional[str]:
    """
    Generate intelligent follow-up questions based on what's missing from the conversation.
    NEVER asks about facts we already know. Adds variety to avoid repetitive questions.

    Args:
        query: Current user query
        grants: List of matched grants
        conversation_history: Recent conversation messages
        known_facts: Dictionary of already-known facts

    Returns:
        Smart follow-up question or None
    """
    import random

    if known_facts is None:
        known_facts = extract_conversation_facts(conversation_history)

    query_lower = query.lower()

    # Check what's missing (only ask about unknowns)
    unknowns = {
        'trl': known_facts.get('trl') is None,
        'clinical': known_facts.get('clinical_champion') is None,
        'timeline': known_facts.get('timeline') is None,
        'budget': known_facts.get('budget') is None,
        'sector': known_facts.get('sector') is None,
        'partnerships': not known_facts.get('partnerships'),
        'revenue': 'revenue' not in str(conversation_history).lower(),
    }

    # Grant-specific strategic questions
    if grants and len(grants) > 0:
        top_grant = grants[0]
        grant_title_lower = top_grant.get('title', '').lower()

        # i4i specific questions
        if 'i4i' in grant_title_lower and unknowns['clinical']:
            clinical_questions = [
                "Who's your clinical champion? NIHR will definitely ask.",
                "Got an NHS trust on board? That's critical for i4i.",
                "Do you have clinical validation lined up? i4i panel obsesses over that.",
            ]
            return random.choice(clinical_questions)

        # Biomedical Catalyst questions
        if 'biomedical catalyst' in grant_title_lower:
            if unknowns['revenue']:
                return "Any revenue yet? Biomedical Catalyst loves to see commercial traction."
            elif unknowns['partnerships']:
                return "Got any industry partners lined up? Helps with Biomedical Catalyst."

        # KTP questions
        if 'ktp' in grant_title_lower or 'knowledge transfer' in grant_title_lower:
            if unknowns['partnerships']:
                return "Which university are you thinking of partnering with for KTP?"
            else:
                return "What specific R&D capability are you looking to access through KTP?"

        # Innovation Loans questions
        if 'loan' in grant_title_lower:
            if unknowns['revenue']:
                return "What's your current revenue? Loans need proof you can repay."
            else:
                return "Can you demonstrate cashflow for loan repayment? Credit team will scrutinize that."

    # General strategic questions (varied)
    if unknowns['trl'] and unknowns['sector']:
        sector_questions = [
            "What sector are you in - medtech, materials, AI?",
            "Quick one - what space are you operating in?",
            "Tell me your sector and rough TRL?",
        ]
        return random.choice(sector_questions)

    if unknowns['trl']:
        trl_questions = [
            "What TRL are you at roughly?",
            "Where are you development-wise - prototype, pilot, or commercial?",
            "What's your tech readiness level looking like?",
        ]
        return random.choice(trl_questions)

    if unknowns['timeline']:
        timeline_questions = [
            "When do you need the funding by?",
            "What's your timeline looking like?",
            "How urgent is this - months or weeks?",
        ]
        return random.choice(timeline_questions)

    if grants and unknowns['budget']:
        first_grant = grants[0]
        funding = first_grant.get('total_fund_gbp') or first_grant.get('total_fund')
        if funding:
            budget_questions = [
                f"Is {funding} the right scale for you?",
                f"Does {funding} match what you need?",
                f"Thinking {funding} range or bigger?",
            ]
            return random.choice(budget_questions)

    # Partnership/commercial questions for depth
    if unknowns['partnerships'] and len(conversation_history) > 3:
        return "Any industry or academic partners on board?"

    return None


def expand_query_for_search(query: str) -> str:
    """
    Expand broad queries with related terms to improve search recall.

    Args:
        query: Original search query

    Returns:
        Expanded query with related terms
    """
    query_lower = query.lower()

    # Comprehensive expansion mappings for better vector search
    expansions = {
        # AI variations
        'ai grant': 'artificial intelligence AI machine learning agentic AI generative AI LLM pioneer',
        'ai funding': 'artificial intelligence AI machine learning agentic AI generative AI',
        'show me ai': 'artificial intelligence AI machine learning agentic generative pioneer',
        'ai': 'artificial intelligence AI machine learning agentic generative',
        'machine learning': 'machine learning ML AI artificial intelligence neural network',
        'llm': 'large language model LLM AI artificial intelligence generative',

        # Biotech/health variations
        'biotech': 'biotechnology biotech life sciences pharmaceutical drug discovery',
        'medtech': 'medical technology medtech medical device healthcare innovation digital health',
        'health': 'healthcare health NHS medical clinical patient care',
        'medical device': 'medical device medtech diagnostic therapeutic healthcare',

        # Business stage
        'startup': 'startup SME small business entrepreneur innovation',
        'sme': 'SME small medium enterprise startup business',
    }

    # Check for expansion matches
    for key, expansion in expansions.items():
        if key in query_lower:
            expanded = f"{query} {expansion}"
            logger.info(f"Expanded query: '{query}' → '{expanded[:100]}...'")
            return expanded

    return query


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns:
        System health status
    """
    # Count documents as a proxy for index size
    # In production, get this from vector index stats
    vector_size = len(vector_index._vectors) if hasattr(vector_index, '_vectors') else 0

    return HealthResponse(
        status="healthy",
        database=DB_PATH,
        vector_index_size=vector_size,
    )


@app.get("/grants", response_model=List[GrantSummary])
async def list_grants(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of grants to return"),
    offset: int = Query(0, ge=0, description="Number of grants to skip"),
    active_only: bool = Query(False, description="Only return active grants"),
):
    """
    List grants with pagination.

    Args:
        limit: Maximum number of results
        offset: Pagination offset
        active_only: Filter for active grants only

    Returns:
        List of grant summaries
    """
    grants = grant_store.list_grants(limit=limit, offset=offset, active_only=active_only)
    return [_grant_to_summary(g) for g in grants]


@app.get("/grants/{grant_id}", response_model=GrantWithDocuments)
async def get_grant(grant_id: str):
    """
    Get detailed information about a specific grant.

    Args:
        grant_id: Grant identifier

    Returns:
        Grant details with associated documents

    Raises:
        HTTPException: If grant not found
    """
    grant = grant_store.get_grant(grant_id)

    if not grant:
        raise HTTPException(status_code=404, detail=f"Grant not found: {grant_id}")

    # Get associated documents
    docs = doc_store.get_documents_for_grant(grant_id)

    return GrantWithDocuments(
        grant=_grant_to_detail(grant),
        documents=[
            DocumentSummary(
                id=d.id,
                doc_type=d.doc_type,
                scope=d.scope,
                source_url=d.source_url,
                length=len(d.text),
            )
            for d in docs
        ],
    )


@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results to return"),
    active_only: bool = Query(True, description="Only return active grants"),
    min_funding: Optional[int] = Query(None, description="Minimum funding in GBP"),
    max_funding: Optional[int] = Query(None, description="Maximum funding in GBP"),
    sources: Optional[List[str]] = Query(None, description="Filter by source (innovate_uk, nihr)"),
    filter_scope: Optional[str] = Query(None, description="Filter by scope: 'competition' or 'global'"),
):
    """
    Perform semantic search over indexed documents with optional filters.

    Args:
        query: Natural language search query
        top_k: Maximum number of results (1-100)
        active_only: Only return active grants
        min_funding: Minimum funding amount in GBP
        max_funding: Maximum funding amount in GBP
        sources: List of sources to include (e.g., ["nihr", "innovate_uk"])
        filter_scope: Optional scope filter

    Returns:
        Search results with relevance scores
    """
    logger.info(f"Search query: {query} (top_k={top_k}, active_only={active_only}, scope={filter_scope})")

    # Over-fetch to account for filtering
    fetch_k = min(top_k * 3, 150)

    # Query vector index
    hits = vector_index.query(
        query_text=query,
        top_k=fetch_k,
        filter_scope=filter_scope,
    )

    # Convert to API schema with filtering
    results: List[SearchHit] = []

    for hit in hits:
        # Get grant details
        grant = grant_store.get_grant(hit.grant_id) if hit.grant_id else None

        if not grant:
            continue

        # Apply filters
        if active_only and not grant.is_active:
            continue

        if min_funding is not None:
            # Parse total_fund to get GBP amount
            fund_amount = None
            if grant.total_fund:
                # Extract numeric value from total_fund string
                import re
                match = re.search(r'[\d,]+', grant.total_fund)
                if match:
                    try:
                        fund_amount = int(match.group().replace(',', ''))
                    except ValueError:
                        pass

            if fund_amount is None or fund_amount < min_funding:
                continue

        if max_funding is not None:
            # Parse total_fund to get GBP amount
            fund_amount = None
            if grant.total_fund:
                import re
                match = re.search(r'[\d,]+', grant.total_fund)
                if match:
                    try:
                        fund_amount = int(match.group().replace(',', ''))
                    except ValueError:
                        pass

            if fund_amount is not None and fund_amount > max_funding:
                continue

        if sources:
            sources_lower = [s.lower() for s in sources]
            if grant.source.lower() not in sources_lower:
                continue

        # Try cached GPT summary first, fallback to chunk snippet
        summary = None
        if hit.grant_id:
            summary = _get_grant_summary(hit.grant_id)

        # Use summary if available, otherwise fallback to snippet
        if summary:
            snippet = summary
        else:
            snippet = _build_snippet(hit.text)

        results.append(
            SearchHit(
                grant_id=hit.grant_id or "unknown",
                title=grant.title,
                source=grant.source,
                score=round(hit.score, 4),
                doc_type=hit.metadata.get("doc_type", "unknown"),
                scope=hit.metadata.get("scope", "unknown"),
                source_url=hit.source_url,
                snippet=snippet,
            )
        )

        # Stop when we have enough results
        if len(results) >= top_k:
            break

    logger.info(f"Search returned {len(results)} results (filtered from {len(hits)} candidates)")

    return SearchResponse(
        query=query,
        total_results=len(results),
        results=results,
    )


@app.post("/search/explain", response_model=ExplainResponse)
async def search_explain(req: ExplainRequest):
    """
    Use GPT-5 to explain which grants best match the query.

    Includes caching to avoid duplicate API calls.

    Process:
    1. Check cache first
    2. Perform semantic search over indexed documents
    3. Retrieve top_k most relevant chunks
    4. Build context from grant details and snippets
    5. Send to GPT-5 for natural language explanation
    6. Cache the result
    7. Return explanation with cited grants

    Args:
        req: Search query and parameters

    Returns:
        GPT-5 generated explanation with grant references

    Raises:
        HTTPException: If GPT-5 client initialization fails or API call fails
    """
    global llm_client

    # Check cache first
    cached = explanation_cache.get(req.query, model="gpt-4o-mini")
    if cached:
        logger.info(f"Returning cached explanation for: {req.query}")
        return ExplainResponse(
            query=req.query,
            explanation=cached["explanation"],
            referenced_grants=[
                ReferencedGrant(**grant)
                for grant in cached["referenced_grants"]
            ],
        )

    # Lazy initialization of GPT-5-mini client
    if llm_client is None:
        try:
            from src.llm.client import LLMClient
            llm_client = LLMClient(model="gpt-4o-mini")
            logger.info("✓ Initialized GPT-5-mini client for /search/explain")
        except ValueError as e:
            logger.error(f"✗ Failed to initialize GPT client: {e}")
            raise HTTPException(
                status_code=500,
                detail="GPT client not configured. Set OPENAI_API_KEY environment variable."
            )
        except Exception as e:
            logger.error(f"✗ Unexpected error initializing GPT: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Perform semantic search
    logger.info(f"Explain query: {req.query} (top_k={req.top_k})")

    hits = vector_index.query(query_text=req.query, top_k=req.top_k * 2)  # Over-fetch

    if not hits:
        logger.info("No relevant grants found")
        return ExplainResponse(
            query=req.query,
            explanation="No relevant grants found for your query. Try broader search terms or different keywords.",
            referenced_grants=[],
        )

    # Group by grant_id and keep best-scoring chunk per grant
    by_grant = {}
    for hit in hits:
        if not hit.grant_id:
            continue
        current = by_grant.get(hit.grant_id)
        if current is None or hit.score > current.score:
            by_grant[hit.grant_id] = hit

    # Sort grants by score, take top N for LLM context
    max_grants_for_llm = 5
    sorted_hits = sorted(by_grant.values(), key=lambda h: h.score, reverse=True)
    selected_hits = sorted_hits[:max_grants_for_llm]

    # Build context from deduplicated grants
    context_blocks: List[str] = []
    referenced_grants: List[ReferencedGrant] = []

    for hit in selected_hits:
        grant = grant_store.get_grant(hit.grant_id)

        if not grant:
            logger.warning(f"Grant not found: {hit.grant_id}")
            continue

        # Extract metadata
        doc_type = hit.metadata.get("doc_type", "unknown")
        scope = hit.metadata.get("scope", "unknown")

        # Get snippet (longer, since we have fewer grants)
        snippet = hit.text[:700] if hasattr(hit, "text") else ""

        # Build context block
        context_blocks.append(
            f"Grant: {grant.title} (source: {grant.source}, id: {grant.id})\n"
            f"Status: {'active' if grant.is_active else 'closed'}\n"
            f"Funding: {grant.total_fund or 'unknown'}\n"
            f"Deadline: {grant.closes_at.isoformat() if grant.closes_at else 'unknown'}\n"
            f"Doc type: {doc_type}, scope: {scope}\n"
            f"Snippet:\n{snippet}\n"
            f"URL: {grant.url}\n"
        )

        referenced_grants.append(
            ReferencedGrant(
                grant_id=grant.id,
                title=grant.title,
                url=grant.url,
                score=round(hit.score, 4),
            )
        )

    logger.info(f"Built context from {len(context_blocks)} unique grants (deduped from {len(hits)} hits)")

    # Improved opinionated prompt
    system_prompt = """You are an expert UK grant funding strategist.

You are given:
- A user query
- A set of candidate grants with snippets and metadata

Your job is to:
- Decide which 1–3 grants are the strongest match.
- Explain your reasoning clearly and efficiently.
- Avoid hedging and fluff.

RULES
- Be opinionated: say which single grant is the top recommendation, if there is one.
- If the matches are weak, say that explicitly and explain why.
- Do NOT invent grants or details that are not supported by the snippets.
- No flattery, no "great question", no long intros. Just answer.

OUTPUT STRUCTURE
1. Short direct answer: 1–2 sentences summarising the overall situation.
2. "Best fits":
   - For each top grant:
     - Name and source (Innovate UK or NIHR)
     - 2–3 bullets on why it fits
     - Funding amount and deadline if available
3. "If you only apply for one…":
   - State which one and why.
4. "Next steps":
   - 2–3 concrete actions (e.g. download specification, check eligibility section, etc.).

Keep it concise. Aim for roughly 400–600 words.
"""

    user_prompt = (
        f"USER QUERY:\n{req.query}\n\n"
        f"RELEVANT GRANTS (ordered by relevance):\n\n" +
        "\n---\n\n".join(context_blocks) +
        "\n\nBased on these grants, explain which ones best match the user's query and why."
    )

    # Call GPT
    try:
        logger.info("Calling GPT...")

        explanation = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,  # Lower temperature for more focused responses
            max_tokens=1500,
        )

        logger.info(f"✓ GPT response received ({len(explanation)} chars)")

    except Exception as e:
        logger.error(f"✗ GPT API call failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )

    # Cache the result
    explanation_cache.set(
        query=req.query,
        explanation=explanation,
        model="gpt-4o-mini",
        referenced_grants=[grant.dict() for grant in referenced_grants],
    )

    return ExplainResponse(
        query=req.query,
        explanation=explanation,
        referenced_grants=referenced_grants,
    )


def normalize_markdown(text: str) -> str:
    """
    Normalize LLM output to ensure clean Markdown formatting.

    Args:
        text: Raw LLM response

    Returns:
        Cleaned Markdown text
    """
    # Ensure headings have blank line before them
    text = re.sub(r'([^\n])(##\s)', r'\1\n\n\2', text)

    # Ensure bullets have proper spacing
    text = re.sub(r'([^\n])(-\s)', r'\1\n\2', text)

    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Prevent accidental bold/heading concatenation
    text = re.sub(r'([A-Za-z])([#*])', r'\1 \2', text)

    return text.strip()


def filter_grants_by_relevance(grant_refs: list, min_score: float = 0.40, max_grants: int = 3) -> list:
    """
    Filter grants to only relevant, high-quality matches.

    Args:
        grant_refs: List of grant references with scores
        min_score: Minimum relevance score threshold
        max_grants: Maximum number of grants to return

    Returns:
        Filtered list of grant references
    """
    # Filter by score
    filtered = [g for g in grant_refs if g.score >= min_score]

    # Sort by score (descending)
    filtered.sort(key=lambda g: g.score, reverse=True)

    # Limit to max_grants
    return filtered[:max_grants]


def classify_grant_matches(
    grouped_grants: list[dict],
    strong_threshold: float = 0.65,
    weak_threshold: float = 0.45,
) -> tuple[list[dict], list[dict]]:
    """
    Classify grants as strong or weak matches based on relevance score.

    Args:
        grouped_grants: List of grant buckets from grant-level aggregation
        strong_threshold: Minimum score for strong match (default 0.65)
        weak_threshold: Minimum score for weak match (default 0.45)

    Returns:
        Tuple of (strong_matches, weak_matches)
    """
    strong = []
    weak = []

    for entry in grouped_grants:
        score = entry["best_score"]
        if score >= strong_threshold:
            strong.append(entry)
        elif score >= weak_threshold:
            weak.append(entry)
        # Below weak_threshold: discard

    return strong, weak


def build_referenced_grants(
    strong_matches: list[dict],
    weak_matches: list[dict],
) -> list[ChatGrant]:
    """
    Build the list of ChatGrant objects for the response,
    marking weak matches as "stretch fit".

    Args:
        strong_matches: List of strong grant buckets
        weak_matches: List of weak grant buckets

    Returns:
        List of ChatGrant objects with stretch_fit flag set appropriately
    """
    chat_grants = []

    # Strong matches first
    for bucket in strong_matches:
        g = bucket["grant"]
        score = bucket["best_score"]

        chat_grants.append(
            ChatGrant(
                grant_id=g.id,
                title=g.title,
                url=g.url,
                source=g.source,
                is_active=g.is_active,
                total_fund_gbp=getattr(g, "total_fund_gbp", None),
                closes_at=g.closes_at.isoformat() if g.closes_at else None,
                score=round(score, 3),
            )
        )

    # Weak matches (stretch fits)
    for bucket in weak_matches:
        g = bucket["grant"]
        score = bucket["best_score"]

        chat_grants.append(
            ChatGrant(
                grant_id=g.id,
                title=g.title,
                url=g.url,
                source=g.source,
                is_active=g.is_active,
                total_fund_gbp=getattr(g, "total_fund_gbp", None),
                closes_at=g.closes_at.isoformat() if g.closes_at else None,
                score=round(score, 3),
            )
        )

    return chat_grants


@app.post("/chat", response_model=ChatResponse)
async def chat_with_grants(req: ChatRequest):
    """
    Chat endpoint with filtered recommendations.

    Uses explain_with_gpt for opinionated, concise responses.
    """
    global chat_llm_client

    query = req.message.strip()
    if not query:
        return ChatResponse(answer="Ask me something about funding.", grants=[])

    logger.info(f"/chat query: {query!r}")

    # Initialize LLM client if needed
    if chat_llm_client is None:
        try:
            from src.llm.client import LLMClient
            chat_llm_client = LLMClient(model="gpt-4o-mini")
            logger.info("✓ Initialized GPT-4o-mini client for /chat")
        except Exception as e:
            logger.error(f"✗ Failed to initialize GPT client: {e}")
            return ChatResponse(
                answer="GPT client not configured. Set OPENAI_API_KEY environment variable.",
                grants=[],
            )

    # Get search hits
    try:
        hits = vector_index.query(
            query_text=query,
            top_k=20,
            filter_scope=None
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return ChatResponse(
            answer="Search failed unexpectedly. Try rephrasing or asking again in a moment.",
            grants=[],
        )

    if not hits:
        return ChatResponse(
            answer="I don't see anything in the current Innovate UK or NIHR data that clearly matches that. "
                   "You might need a different funding body or a more general innovation grant.",
            grants=[],
        )

    # Generate response with filtering
    try:
        answer_markdown, recommended_grants = explain_with_gpt(
            chat_llm_client,
            query,
            hits
        )
    except Exception as e:
        logger.error(f"GPT explanation failed: {e}")
        return ChatResponse(
            answer="I found relevant grants, but the AI layer failed while drafting the explanation. "
                   "Try asking again or narrow your question.",
            grants=[],
        )

    # Build response - convert recommended_grants to ChatGrant objects
    grant_refs = []
    for g in recommended_grants:
        grant_refs.append(
            ChatGrant(
                grant_id=g.get("grant_id", ""),
                title=g.get("title", ""),
                url=g.get("url", "#"),
                source=g.get("source", ""),
                is_active=True,  # Default since we're filtering by active
                total_fund_gbp=g.get("total_fund_gbp"),
                closes_at=g.get("closes_at"),
                score=g.get("best_score", 0.0)
            )
        )

    return ChatResponse(answer=answer_markdown, grants=grant_refs[:5])


@app.post("/chat/stream")
async def chat_with_grants_stream(req: ChatRequest):
    """
    Streaming chat endpoint with real-time responses.

    Returns Server-Sent Events with JSON chunks:
    - {"type": "token", "content": "..."}  - Streamed response text
    - {"type": "grants", "grants": [...]}  - Recommended grants at the end
    - {"type": "done"}                      - End of stream
    """
    import json

    global chat_llm_client

    query = req.message.strip()
    if not query:
        async def empty_generator():
            yield f"data: {json.dumps({'type': 'token', 'content': 'Ask me something about funding.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(empty_generator(), media_type="text/event-stream")

    logger.info(f"/chat/stream query: {query!r}")

    # Validate TRL if mentioned - catch invalid values early
    trl_validation = validate_and_correct_trl(query)
    if not trl_validation['valid'] and trl_validation['correction']:
        # User provided invalid TRL - correct them immediately
        logger.warning(f"Invalid TRL detected in query: {query}")
        async def trl_correction_generator():
            yield f"data: {json.dumps({'type': 'token', 'content': trl_validation['correction']})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(trl_correction_generator(), media_type="text/event-stream")

    # Initialize LLM client if needed
    if chat_llm_client is None:
        try:
            from src.llm.client import LLMClient
            chat_llm_client = LLMClient(model="gpt-4o-mini")
            logger.info("✓ Initialized GPT-4o-mini client for /chat/stream")
        except Exception as e:
            logger.error(f"✗ Failed to initialize GPT client: {e}")
            async def error_generator():
                yield f"data: {json.dumps({'type': 'token', 'content': 'GPT client not configured. Set OPENAI_API_KEY environment variable.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return StreamingResponse(error_generator(), media_type="text/event-stream")

    # Generate streaming response
    async def generate():
        try:
            # Step 0: Check if message contains a URL and parse company website
            url = extract_url_from_message(query)
            website_context = ""
            enhanced_query = query

            if url:
                logger.info(f"🌐 Detected website URL: {url}")
                try:
                    company_info = parse_company_website(url)

                    if company_info:
                        # Build rich, actionable context
                        keywords_str = ', '.join(company_info['keywords'][:5]) if company_info['keywords'] else 'tech company'
                        grants_str = ', '.join(company_info.get('suggested_grants', [])) if company_info.get('suggested_grants') else 'Not determined'

                        website_context = f"""

🌐 ANALYZED {url}:
Company: {company_info['title']}
Sector: {company_info.get('sector', 'general')}
Estimated TRL: {company_info.get('trl_estimate', 'unknown')}
Stage: {company_info['stage']}
Tech focus: {keywords_str}
Likely grant fits: {grants_str}

IMPORTANT: Use this intel to give SPECIFIC advice tailored to THIS company. Reference their actual sector and stage.
Example: "Just checked out {company_info['title']} - {company_info.get('sector', 'your')} platform, right? For {company_info.get('sector', 'your sector')} at TRL {company_info.get('trl_estimate', 'your stage')}, here's what makes sense..."
"""

                        # Enhance query with sector and keywords for better search
                        search_terms = []
                        if company_info.get('sector') and company_info['sector'] != 'general':
                            search_terms.append(company_info['sector'])
                        if company_info.get('keywords'):
                            search_terms.extend(company_info['keywords'][:3])

                        if search_terms:
                            enhanced_query = f"{query} {' '.join(search_terms)}"
                            logger.info(f"✓ Enhanced query: {enhanced_query[:100]}...")
                        else:
                            logger.warning(f"⚠️ No search terms extracted from {url}")
                    else:
                        logger.warning(f"⚠️ parse_company_website returned None for {url}")
                        website_context = f"\n🌐 Note: User provided website {url} but couldn't parse it automatically.\n"

                except Exception as e:
                    logger.error(f"❌ Failed to parse website {url}: {e}")
                    import traceback
                    traceback.print_exc()
                    website_context = f"\n🌐 Note: User provided website {url} but couldn't parse it automatically.\n"

            # Step 1: Detect if user wants MORE grants (not details about same grant)
            query_lower = query.lower()
            expand_search_indicators = [
                'anything else', 'what else', 'show me more', 'other grants',
                'more options', 'what about other', 'are there others',
                'alternative', 'different grant', 'other opportunities'
            ]
            wants_more_grants = any(indicator in query_lower for indicator in expand_search_indicators)

            # Step 2: Resolve references and follow-ups with context locking
            resolved_query = enhanced_query
            reference_detected = False
            locked_on_grant = False  # Track if we're locked onto a specific grant
            grant_name = None  # Initialize to avoid undefined variable errors
            last_grants_mentioned = []
            most_recent_grant = None  # Track the most recently discussed grant
            max_grants_to_show = 3  # Default

            if wants_more_grants:
                # User wants to see MORE grants, not details about the current one
                logger.info("User requesting more grant options - expanding search")

                # Look back to find the ORIGINAL topic/query (not "show me more" queries)
                original_query = None
                if req.history:
                    for msg in reversed(req.history):
                        if msg.role == "user":
                            msg_lower = msg.content.lower()
                            # Skip if it's also a "show me more" type query
                            if not any(ind in msg_lower for ind in expand_search_indicators):
                                original_query = msg.content
                                logger.info(f"✓ Found original query: '{original_query}'")
                                break

                if original_query:
                    # Keep the original topic context
                    resolved_query = original_query
                    logger.info(f"Expanding search based on original topic: '{resolved_query}'")
                else:
                    # Fallback to enhanced query
                    resolved_query = enhanced_query
                    logger.info(f"No original query found, using: '{resolved_query}'")

                locked_on_grant = False  # DON'T lock
                max_grants_to_show = 5  # Show more options

            # Extract grants from recent conversation history
            elif req.history:
                last_assistant = None
                grant_discussion_depth = 0

                # Look through recent history to find grants being discussed
                for msg in reversed(req.history[-6:]):  # Last 3 exchanges
                    if msg.role == "assistant":
                        # SKIP welcome messages and short UI responses
                        if len(msg.content) < 100:
                            logger.info(f"Skipping short assistant message (len={len(msg.content)})")
                            continue

                        # SKIP if it's the generic welcome
                        if "Hi, I'm Ailsa" in msg.content or "I can help you discover" in msg.content:
                            logger.info("Skipping welcome message")
                            continue

                        if not last_assistant:
                            last_assistant = msg.content

                        grant_discussion_depth += 1

                        # Pattern 1: Grant names with common keywords (Award, Grant, Partnership, Programme, etc.)
                        grant_matches = re.findall(
                            r'^(?:#+\s*)?([A-Z][^\n]{15,100}(?:Grant|Award|Partnership|Programme|Catalyst|Fund|Loan|Competition|Prize)[^\n]{0,30}?)(?:\n|$)',
                            msg.content,
                            re.MULTILINE | re.IGNORECASE
                        )

                        # Pattern 2: Extract from structured responses (before colons or from headers)
                        if not grant_matches:
                            grant_matches = re.findall(r'(?:^|\n)(?:###?\s+)?([A-Z][^\n:]{10,80})(?:\n|:)', msg.content, re.MULTILINE)

                        if grant_matches:
                            cleaned = [g.strip() for g in grant_matches[:3]]
                            if not last_grants_mentioned:
                                last_grants_mentioned = cleaned
                            # Track the most recently mentioned grant
                            if not most_recent_grant:
                                most_recent_grant = cleaned[0]
                                logger.info(f"Grant being discussed: {most_recent_grant} (depth: {grant_discussion_depth})")

                        # Break after processing first assistant message for numbered items
                        if grant_discussion_depth >= 2:  # Look at last 2 assistant messages
                            break

                # 1A. Handle numbered references (e.g., "number 2", "#3", "the first one")
                if last_assistant and re.search(r'\b(?:number|#|first|second|third|1st|2nd|3rd)\s*[1-5]?\b', query.lower()):
                    # Extract numbered items (formats: "1. Grant Name", "**1. Grant Name**", etc.)
                    numbered_items = re.findall(r'\d+\.\s*\*?\*?([^\n:*]+)', last_assistant)

                    # Find which number user is asking about
                    number_match = re.search(r'(?:number|#)\s*(\d+)', query.lower())
                    if not number_match:
                        # Handle ordinal words
                        word_to_num = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}
                        for word, num in word_to_num.items():
                            if word in query.lower():
                                number_match = type('obj', (object,), {'group': lambda x, n=num: n})()
                                break

                    if number_match and numbered_items:
                        try:
                            num = int(number_match.group(1))
                            if 1 <= num <= len(numbered_items):
                                grant_name = numbered_items[num - 1].strip()
                                resolved_query = f"{query} {grant_name}"
                                reference_detected = True
                                logger.info(f"✓ Resolved numbered reference: '{query}' → added '{grant_name}'")
                        except (ValueError, AttributeError):
                            pass

                # 1B. Detect follow-up questions with typo-tolerant pronoun detection
                if not reference_detected and most_recent_grant:
                    query_lower = query.lower()

                    # Comprehensive follow-up indicators (including typos)
                    follow_up_indicators = [
                        # Direct references
                        'this opportunity', 'that opportunity', 'this grant', 'that grant',
                        'the grant', 'this one', 'that one', 'the opportunity',
                        'the award', 'the programme', 'the program', 'the fund', 'the loan',
                        'the competition', 'the prize',

                        # Common typos for "this"
                        'thus', 'thos', 'thsi', 'thid', 'thius',

                        # Standalone pronouns
                        'it ', 'this ', 'that ',

                        # Implicit follow-ups (questions that only make sense with context)
                        'application questions', 'application process', 'application form',
                        'how do i apply', 'how to apply', 'apply for',
                        'eligibility', 'who can apply', 'what are the requirements',
                        'requirements for', 'criteria for',
                        'deadline', 'when is it due', 'when does it close',
                        'how much funding', 'funding amount',
                        'next steps', 'what should i do', 'how does',
                        'what does', 'tell me more', 'more about'
                    ]

                    # Check if this is a follow-up question
                    is_follow_up = any(indicator in query_lower for indicator in follow_up_indicators)

                    # Also check if query starts with question words without context (implicit follow-up)
                    question_starters = ['what are', 'what is', 'how do', 'how does', 'when is', 'who can']
                    starts_with_question = any(query_lower.startswith(q) for q in question_starters)

                    if is_follow_up or (starts_with_question and len(query.split()) < 8):
                        # LOCK ONTO the grant being discussed
                        grant_name = most_recent_grant
                        resolved_query = f"{most_recent_grant} {query}"
                        reference_detected = True
                        locked_on_grant = True
                        logger.info(f"🔒 LOCKED onto grant: {most_recent_grant}")
                        logger.info(f"✓ Follow-up detected: '{query}' → '{resolved_query}'")

                # 1C. Handle vague follow-ups (e.g., "tell me more", "application process", "how do I apply")
                if not reference_detected and last_grants_mentioned:
                    followup_patterns = [
                        "tell me more", "more about", "application process", "how do i apply",
                        "how to apply", "what about", "eligibility", "who can apply",
                        "deadline", "next steps", "more details", "more info",
                        "application questions", "application form", "how does", "what does"
                    ]

                    if any(pattern in query.lower() for pattern in followup_patterns):
                        # They're asking about the grant we just discussed
                        grant_name = last_grants_mentioned[0] if last_grants_mentioned else most_recent_grant
                        if grant_name:
                            resolved_query = f"{query} {grant_name}"
                            reference_detected = True
                            logger.info(f"✓ Detected follow-up: '{query}' → added '{grant_name}'")

            # Step 2: Check if we have real conversation history (not just welcome message)
            has_real_history = False
            if req.history:
                for msg in req.history:
                    if msg.role == "assistant" and len(msg.content) > 100 and "Hi, I'm Ailsa" not in msg.content:
                        has_real_history = True
                        break

            # Step 3: Contextualize query using conversation history (if not already resolved)
            if not reference_detected and has_real_history and req.history:
                # Build context from recent messages, excluding welcome messages
                recent_context = " ".join([
                    f"{msg.content[:100]}"
                    for msg in req.history[-4:]  # Last 2 exchanges
                    if msg.role == "assistant" and len(msg.content) > 100 and "Hi, I'm Ailsa" not in msg.content
                ])

                # If the query is short/vague, enrich it with context
                if len(query.split()) < 10 and recent_context:
                    resolved_query = f"{query} {recent_context}"
                    logger.info(f"Contextualized short query with history: {resolved_query[:150]}...")
            elif not has_real_history:
                logger.info(f"Fresh query (no real history): '{query}'")

            # Step 4: Expand query for better vector search
            search_query = expand_query_for_search(resolved_query)

            # Step 5: Perform vector search with expanded query
            logger.info(f"Vector search query: '{search_query[:100]}...'")
            try:
                hits = vector_index.query(
                    query_text=search_query,
                    top_k=20,
                    filter_scope=None
                )
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                yield f"data: {json.dumps({'type': 'token', 'content': 'Search failed unexpectedly. Try rephrasing or asking again in a moment.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            logger.info(f"Vector search returned {len(hits)} hits")
            if hits:
                logger.info(f"Top 3 scores: {[f'{h.score:.3f}' for h in hits[:3]]}")
                logger.info(f"Top hit: {hits[0].grant_id if hasattr(hits[0], 'grant_id') else 'unknown'}")

            # CRITICAL: Filter out Smart Grants (paused January 2025)
            hits_before_filter = len(hits)
            filtered_hits = []
            for h in hits:
                # Get grant details to check title
                grant = grant_store.get_grant(h.grant_id) if h.grant_id else None
                if grant:
                    title_lower = grant.title.lower()
                    if "smart grant" in title_lower or "smart grants" in title_lower:
                        logger.info(f"🚫 Filtered out Smart Grant: {grant.title}")
                        continue
                filtered_hits.append(h)

            hits = filtered_hits
            if hits_before_filter != len(hits):
                logger.info(f"Filtered out {hits_before_filter - len(hits)} Smart Grants, {len(hits)} hits remaining")

            if not hits:
                msg = "I don't see anything in the current Innovate UK or NIHR data that clearly matches that. You might need a different funding body or a more general innovation grant."
                yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # Step 6: Select top grants using resolved/contextualized query
            # Use dynamic max_grants based on whether user wants more
            from collections import defaultdict

            # First, get all potential grants
            all_grants = select_top_grants(hits, query=resolved_query)

            # If user wants more grants, filter out already discussed ones
            grants = all_grants
            if wants_more_grants and req.history:
                already_discussed_titles = set()
                for msg in req.history:
                    if msg.role == "assistant":
                        # Extract grant titles from previous responses
                        discussed = re.findall(
                            r'(?:^|\n)(?:###?\s+)?([^\n]{15,100}(?:Grant|Award|Prize|Programme|Catalyst|Fund|Loan|Competition))',
                            msg.content,
                            re.MULTILINE | re.IGNORECASE
                        )
                        already_discussed_titles.update([d.strip() for d in discussed])

                # Filter out already discussed grants
                grants_before = len(grants)
                grants = [g for g in grants if g.get('title') not in already_discussed_titles]
                logger.info(f"Filtered out {grants_before - len(grants)} already-discussed grants from {len(already_discussed_titles)} total discussed")

            # Limit to max_grants_to_show
            grants = grants[:max_grants_to_show]

            logger.info(f"Selected {len(grants)} grants to recommend (max: {max_grants_to_show})")
            if not grants:
                logger.error(f"⚠️ ZERO GRANTS SELECTED for query: '{query}'")
                logger.error(f"Original hits: {len(hits)}")
                if hits:
                    logger.error(f"Top hit was: grant_id={hits[0].grant_id if hasattr(hits[0], 'grant_id') else 'unknown'}, score={hits[0].score:.3f}")

            # Step 7: Build context
            context = build_llm_context(query, hits, grants)

            # Step 7a: Extract known facts from conversation to avoid asking twice
            known_facts = extract_conversation_facts(req.history or [])
            logger.info(f"Known facts: {known_facts}")

            # Step 7b: Analyze query intent for better context
            query_intent = analyze_query_intent(query)
            logger.info(f"Query intent: {query_intent}")

            # Step 7c: Generate smart follow-up if appropriate (uses known_facts)
            smart_followup = get_smart_followup(query, grants, req.history or [], known_facts)
            if smart_followup:
                logger.info(f"Generated smart follow-up: {smart_followup}")

            # Step 6: Load expert examples dynamically
            try:
                expert_examples = get_expert_examples(limit=3, min_quality=4)
                expert_examples_text = format_expert_examples_for_prompt(expert_examples)
                logger.info(f"Loaded {len(expert_examples)} expert examples for prompt")
            except Exception as e:
                logger.warning(f"Failed to load expert examples: {e}")
                expert_examples_text = ""

            # Step 7: Create streaming-specific prompt (outputs MARKDOWN, not JSON!)
            grants_list = "\n".join([f"- {g['title']} ({g['source']})" for g in grants[:5]])

            # Build conversation context for the system prompt
            conversation_context = ""
            original_topic_for_prompt = None

            if req.history:
                conversation_context = "\n\nRECENT CONVERSATION:\n"

                # Find the original topic for context
                for msg in reversed(req.history):
                    if msg.role == "user":
                        msg_lower = msg.content.lower()
                        if not any(ind in msg_lower for ind in expand_search_indicators):
                            original_topic_for_prompt = msg.content
                            break

                # Show recent messages
                for msg in req.history[-4:]:  # Last 2 exchanges
                    role = "User" if msg.role == "user" else "You (Ailsa)"
                    content = msg.content[:300]  # Increased to capture more context
                    conversation_context += f"{role}: {content}...\n"

                # Add original topic context if we're showing more grants
                if wants_more_grants and original_topic_for_prompt:
                    conversation_context += f"\n💡 ORIGINAL TOPIC: {original_topic_for_prompt}\n"
                    conversation_context += "User is asking for MORE options related to this topic. Show additional relevant grants.\n"

                # Add special alert for reference queries and context locking
                reference_keywords = ["number", "#", "first", "second", "third", "that grant", "it", "the one", "this one", "those"]
                if any(keyword in query.lower() for keyword in reference_keywords):
                    conversation_context += "\n⚠️ USER IS REFERENCING SOMETHING FROM YOUR PREVIOUS RESPONSE - check the conversation above.\n"
                    if reference_detected and grant_name:
                        conversation_context += f"✓ Reference resolved: User is asking about '{grant_name}'\n"

                # Add context lock indicator when locked onto a specific grant
                if locked_on_grant and grant_name:
                    conversation_context += f"\n🔒 CONTEXT LOCKED: User is asking specifically about '{grant_name}'\n"
                    conversation_context += f"DO NOT switch to other grants. Answer ONLY about {grant_name}.\n"
                    conversation_context += f"The search results below are filtered for {grant_name}.\n"

            EXPERT_SYSTEM_PROMPT = f"""You are an experienced UK grant consultant. Talk naturally but stay focused.

VOICE RULES:
- Be conversational but concise - aim for 2-3 solid paragraphs max
- Skip the war stories unless genuinely relevant
- If you know something (TRL, budget, sector), NEVER ask again - reference it instead
- Lead with the answer, then ask ONE strategic follow-up question
- No rambling, no repetition, no unnecessary context

RESPONSE STRUCTURE:
1. Direct answer to their question (1-2 sentences)
2. Key insight or warning (1-2 sentences)
3. ONE strategic question to move forward

GOOD EXAMPLES:

"TRL 4 puts you in the sweet spot for i4i actually. You'll need clinical validation though - got a clinical lead?"

"Since you're at TRL 4 in materials, the ATI Programme is your best bet - £1.5M available. What's the application - aerospace or automotive?"

BAD EXAMPLES (Never do this):

"Oh that's interesting! TRL 4, I see. Well, I remember working with another company at TRL 4 and they had quite a journey..."

"There are 3 relevant grants: 1. NIHR i4i Product Development Award... 2. Biomedical Catalyst... 3. Smart Grants..."

CONVERSATION MEMORY - CRITICAL:
When facts are established, reference them, don't re-ask:
✅ "Since you're at TRL 4..."
❌ "What's your TRL?"

✅ "Given your £500k budget..."
❌ "How much funding do you need?"

ALREADY KNOWN FACTS - DO NOT ASK ABOUT THESE AGAIN:
{format_known_facts(known_facts)}

GRANT-SPECIFIC INSIDER KNOWLEDGE (weave in naturally when relevant):

NIHR i4i:
- Obsessed with clinical pull vs technology push - reviewers hate "cool tech looking for a problem"
- MUST have NHS/clinical champion on board - non-negotiable
- Patient benefit needs to be crystal clear in first paragraph
- Information Governance critical for any data-related projects
- Success rate ~15-20%, very competitive
- Takes 6-9 months from application to decision

Biomedical Catalyst:
- Early stage = feasibility studies (smaller £)
- Late stage = development to market (larger £)
- Commercial route must be clearly mapped out
- Match funding helps but not required
- Combines Innovate UK commercial focus + NIHR clinical rigor

Innovation Loans:
- Repayable: 3.7% during project, 7.4% during repayment
- Credit team scrutinizes financials HARD - need proof of repayment capacity
- Good for later-stage companies with revenue
- More flexible than grants but riskier financially
- Don't apply if you can't demonstrate cashflow

Knowledge Transfer Partnership (KTP):
- 2-3 year projects partnering with universities
- You get skilled graduate + academic expertise
- Government covers ~2/3 of costs
- Perfect for accessing university facilities/knowledge you don't have
- Great way to de-risk R&D

Smart Grants:
- PAUSED as of January 2025, not accepting applications
- If asked: "Smart Grants are on hold. Try Biomedical Catalyst or sector-specific competitions instead."

CONVERSATIONAL EXAMPLES (how to actually sound):

Query: "What grants for medical devices?"
✅ Good: "For medtech, i4i is your main play - but they want clinical evidence, not just a cool device. Got an NHS trust interested?"
❌ Bad: "Medical device grants include: NIHR i4i Product Development Award, Biomedical Catalyst, Innovation Loans..."

Query: "How much can I get?"
✅ Good: "i4i typically goes up to £1M, sometimes £2M for really strong projects. What's your budget looking like?"
❌ Bad: "Funding amounts vary by grant type. Please see the grant cards below for specific amounts."

Query: "Should I apply for a loan or grant?"
✅ Good: "Grants are free money but 10-20% success rates. Loans give you more control at 3.7% rates, but the Credit team will grill your financials. What's your runway?"
❌ Bad: "Both options have merits. Grants provide non-repayable funding while loans offer flexibility with structured repayment."

Query: "When's the deadline?"
✅ Good: "i4i closes December 3rd - that's 22 days from now. Doable if you start this week, but it'll be tight."
❌ Bad: "The deadline for the NIHR i4i Product Development Award Round 29 is December 3rd, 2024."

CRITICAL: Output PLAIN MARKDOWN only (NOT JSON!). This will be displayed directly to users as it streams.
{conversation_context}
{website_context}

{expert_examples_text}

---EXPERT RESPONSE STYLE---

Learn from these examples of how expert funding advisors respond to clients:

EXAMPLE 1: Positioning for a procurement opportunity
User: Client working on wills/probate tech
Query: Identifying relevant procurement opportunities

Response style:
"Tender for Ministry of Justice Probate Document Management

Ailsa's Take: This tender's requested scope around physical document management is a little unimaginative – however, it's best treated as a clear signal that the Ministry of Justice is committing resources to handling the problem.

Why it's a fit: I would suggest you get in touch directly with the procurement manager for this call (asap while he is still processing the PMEQs, but certainly ahead of the tender officially launching next year December 2026) and convert it into a pitch to discuss how your solution could facilitate these or adjacent needs for the Ministry of Justice. Get in early, boldly, and transform their needs.

Next steps: Contact the procurement manager with a brief customised pitch"

EXAMPLE 2: Explaining loan mechanisms with honest assessment
User: Early-stage company considering innovation loans
Query: Feasibility of loan funding

Response style:
"Innovate UK Innovation Loans Round 24

Ailsa's Take: This government-backed loan mechanism supports late stage (TRL6+) development of innovative new products significantly ahead of others currently available, with a clear route to commercialisation. Loans of £100K-£5M over maximum 5 years at 3.7% interest during project period and 7.4% during repayment.

Why it's a fit: We can make a solid case for innovation. The biggest go/no-go factor (apart from this being a repayable loan rather than a grant) is around fiscal position and ability to demonstrate loan repayment capacity. Innovate UK have a dedicated Credit team who assess debt service coverage and liquidity ratios in addition to the typical assessor panel."

EXAMPLE 3: Feasibility grant with specific success metrics
User: Materials innovation company
Query: Grant opportunities for feasibility studies

Response style:
"Innovate UK: National Materials Innovation Programme: Feasibility Studies

Ailsa's Take: A nice tidy grant for £50k-100k feasibility study exercises, with strong anticipated success of 20% – 7.5X better than the last Innovate UK Smart grant round. Projects are capped at 9 months and start by 1st May 2026, with 30% max subcontractor usage.

Why it's a fit: Projects must focus on 1 of 7 themes. Framing will be key here to align with the right theme. They will not fund projects focused on chemical synthesis rather than adoption of materials innovations in end applications.

Next steps: Secure letter of support by [deadline]; application drafting begins by [date]"

---KEY PATTERNS TO FOLLOW---

STRUCTURE (when discussing specific grants):
- Grant/Opportunity Title (clear and specific)
- "Ailsa's Take:" - Strategic assessment with insider knowledge
- "Why it's a fit:" - Client-specific relevance and positioning advice
- "Next steps:" - Concrete, actionable steps with deadlines when known

TONE & LANGUAGE:
- Confident and directive: "I would suggest", "Get in early, boldly"
- Insider knowledge: Reference TRL levels, success rates, program themes, specific criteria
- Honest about constraints: "The biggest go/no-go factor", "will not fund"
- Specific numbers: "£50k-100k", "20% success rate", "9 months", "3.7% interest"
- Strategic framing: "transform their needs", "position for", "snug fit"
- Natural phrases: "nice tidy grant", "great foot in the door", "snug fit"

WHAT TO INCLUDE:
- Actual funding amounts and ranges
- Deadlines and timelines
- Success rates when known
- Specific eligibility criteria or constraints
- Strategic positioning advice (not just "you're eligible")
- Risk factors or deal-breakers
- Concrete next steps

WHAT TO AVOID:
- Generic enthusiasm without specifics
- Hedging when you have conviction ("This could work" vs "This is a snug fit")
- Lists of requirements without strategic commentary
- Vague advice without actionable steps

ADAPT STRUCTURE BASED ON QUERY:
- Specific grant inquiry → Use full "Ailsa's Take / Why it's a fit / Next steps" structure
- General questions → Conversational but maintain confident, specific tone
- Eligibility questions → Honest assessment with specific criteria
- Strategy questions → Bold, directive positioning advice

Remember: You're a strategic advisor with insider knowledge, not just an information source. Transform needs, position boldly, and be specific with numbers and timelines.

---END EXPERT STYLE GUIDE---

CONVERSATIONAL MEMORY:
- Remember what you JUST discussed - users expect continuity
- If you just talked about i4i and they say "What's the deadline?", they mean i4i
- If you mentioned 3 grants and they say "Tell me about the second one", you know what they mean
- Don't make them repeat themselves - that's annoying in real conversations

HANDLING FOLLOW-UPS NATURALLY:
- If you just talked about Biomedical Catalyst and they ask "What's the deadline?", they mean Biomedical Catalyst
- If they say "Tell me about the second one", you know they mean #2 from your list
- When they ask "How do I apply?", assume they mean the grant you JUST discussed
- Don't hedge or apologize - just answer: "For Biomedical Catalyst, the deadline is..." not "I'm not sure which grant you mean..."

WHEN THEY WANT MORE:
- "Anything else?" = Show different grants on the same topic
- "What about the other one?" = Switch focus to a different grant from your list
- Be honest if you've shown the best: "Honestly, those are your top matches. The rest are pretty marginal for what you need."

SMART GRANTS ARE PAUSED:
- Don't recommend Smart Grants - they're paused as of January 2025
- If someone asks about them: "Smart Grants are on hold right now. For similar funding, look at Biomedical Catalyst or sector-specific competitions instead."

HOW TO ACTUALLY TALK:
Instead of: "There are 3 relevant grants: 1. NIHR i4i Product Development..."
Say: "For a medtech at your stage, I'd start with i4i - it's NIHR's main product development stream. Have you got clinical evidence yet? That's the big one they care about."

Instead of: "## Eligibility Requirements\n- UK registered company\n- TRL 4-6"
Say: "Quick check - you're UK-based and somewhere around TRL 4-6, right? Because if you're earlier than that, i4i probably isn't the move."

Instead of: "The deadline for this opportunity is January 15, 2025."
Say: "Deadline's January 15th - so you've got about a month. Doable if you start this week."

The grants shown below your response:
{grants_list}

Remember: You're chatting with a founder over coffee, not writing a grant database entry. Be helpful, direct, and real."""

            # Build enhanced user prompt with intent and follow-up guidance
            user_content = f"USER QUERY: {query}\n"
            user_content += f"Query Intent: {query_intent}\n\n"

            # Add conversation context if available
            if req.history and len(req.history) > 0:
                recent_exchanges = "\n".join([
                    f"{'User' if msg.role == 'user' else 'You'}: {msg.content[:150]}..."
                    for msg in req.history[-3:]  # Last 3 messages
                    if len(msg.content) > 50  # Skip short/welcome messages
                ])
                if recent_exchanges:
                    user_content += f"RECENT CONVERSATION:\n{recent_exchanges}\n\n"

            user_content += f"RELEVANT GRANT CONTEXT:\n{context}\n\n"

            # Add smart follow-up guidance if we generated one
            if smart_followup:
                user_content += f"💡 Consider ending with this follow-up: {smart_followup}\n\n"

            user_content += "Respond naturally like you're their experienced advisor. Be conversational, direct, and helpful."

            messages = [
                {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            # Adjust temperature based on conversation dynamics
            conversation_length = len(req.history) if req.history else 0
            dynamic_temp = adjust_temperature_for_conversation(conversation_length, len(query))
            logger.info(f"Dynamic temperature: {dynamic_temp} (conv_len={conversation_length}, query_len={len(query)})")

            # Initialize grant mention detector
            grant_detector = GrantMentionDetector(grant_store)
            full_response = ""

            # Stream the response tokens as they arrive from LLM
            for chunk in chat_llm_client.chat_stream(
                messages=messages,
                temperature=dynamic_temp,  # Dynamically adjusted based on conversation context
                max_tokens=400,   # Shorter, punchier responses (2-3 paragraphs)
            ):
                full_response += chunk
                # Stream each token immediately to the client
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            # Auto-detect grant mentions in the response
            mentioned_grants = grant_detector.extract_grant_mentions(full_response, grants)
            logger.info(f"Detected {len(mentioned_grants)} grant mentions in response")

            # Prepare grant cards - prioritize mentioned grants, then show search results
            grant_refs = []
            seen_ids = set()

            # First add mentioned grants (high priority)
            for g in mentioned_grants:
                if g.get('grant_id') not in seen_ids:
                    seen_ids.add(g['grant_id'])
                    grant_refs.append(g)

            # Then add remaining search result grants (if not already included)
            for g in grants[:5]:
                if g.get('grant_id') not in seen_ids:
                    seen_ids.add(g['grant_id'])
                    grant_refs.append(g)

            # Enrich grant cards with full metadata
            enriched_grant_refs = []
            for g in grant_refs[:5]:  # Limit to top 5
                # Get full grant details for enrichment
                try:
                    full_grant = grant_store.get_grant(g["grant_id"])
                    if full_grant:
                        enriched_grant_refs.append({
                            "grant_id": g["grant_id"],
                            "title": g["title"],
                            "url": g["url"],
                            "source": g["source"],
                            "is_active": full_grant.is_active,
                            "total_fund_gbp": full_grant.total_fund_gbp if hasattr(full_grant, 'total_fund_gbp') else None,
                            "closes_at": full_grant.closes_at.isoformat() if full_grant.closes_at else None,
                            "score": g.get("best_score", 0.95)  # High score for mentioned grants
                        })
                    else:
                        # Fallback without enrichment
                        enriched_grant_refs.append({
                            "grant_id": g["grant_id"],
                            "title": g["title"],
                            "url": g["url"],
                            "source": g["source"],
                            "is_active": True,
                            "total_fund_gbp": g.get("total_fund_gbp"),
                            "closes_at": g.get("closes_at"),
                            "score": g.get("best_score", 0.95)
                        })
                except Exception as e:
                    logger.warning(f"Failed to enrich grant {g['grant_id']}: {e}")
                    # Use basic info
                    enriched_grant_refs.append({
                        "grant_id": g["grant_id"],
                        "title": g.get("title", "Unknown Grant"),
                        "url": g.get("url", "#"),
                        "source": g.get("source", "unknown"),
                        "is_active": True,
                        "total_fund_gbp": None,
                        "closes_at": None,
                        "score": g.get("best_score", 0.95)
                    })

            # Send grants (mentioned grants prioritized)
            yield f"data: {json.dumps({'type': 'grants', 'grants': enriched_grant_refs})}\n\n"

            # Done
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'token', 'content': 'I found relevant grants, but encountered an error generating the response. Try asking again or narrow your question.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# -----------------------------------------------------------------------------
# Startup Event
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """
    Run on application startup.

    Load existing documents into vector index.
    """
    logger.info("=" * 80)
    logger.info("Grant Discovery API - Starting")
    logger.info("=" * 80)
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Embeddings: text-embedding-3-small (OpenAI)")
    logger.info(f"LLM: GPT-4o-mini via OpenAI (lazy init)")
    logger.info(f"Docs: http://localhost:8000/docs")
    logger.info("=" * 80)

    # Load all grants and documents into vector index
    try:
        grants = grant_store.list_grants(limit=1000)
        logger.info(f"Found {len(grants)} grants in database")

        total_docs = 0
        for grant in grants:
            docs = doc_store.get_documents_for_grant(grant.id)
            if docs:
                vector_index.index_documents(docs)
                total_docs += len(docs)

        logger.info(f"Loaded {total_docs} documents into vector index")
    except Exception as e:
        logger.error(f"Error loading documents into vector index: {e}")

    logger.info("=" * 80)
