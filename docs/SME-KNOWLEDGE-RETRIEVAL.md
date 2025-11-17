# SME Knowledge Retrieval Fix

## Problem

User added TRL URL to Slack, the bot ingested it into the database, but when asking "is there a link to TRL levels", the system didn't retrieve it from the expert knowledge.

**Root Cause**: Chat endpoints (`/chat` and `/chat/stream`) only used grant context, not SME expert knowledge from the `expert_examples` table.

## Solution

Integrated `search_expert_knowledge()` function into both chat endpoints to search the expert_examples table and include relevant SME knowledge in the LLM context.

## Implementation Details

### 1. Search Function: `search_expert_knowledge()`

Location: [server.py:92-164](../src/api/server.py#L92-L164)

**Features**:
- Hybrid keyword search across `expert_examples` table
- Synonym expansion (e.g., 'trl' → 'technology', 'readiness', 'level')
- URL content boosting when user asks for links
- Relevance scoring based on keyword matches
- Returns top N results formatted for LLM context

**Example**:
```python
expert_knowledge = search_expert_knowledge("is there a link to TRL levels", limit=3)
# Returns: TRL URL from UKRI with full context
```

### 2. Integration in `/chat` Endpoint

Location: [server.py:787-802](../src/api/server.py#L787-L802)

```python
# Step 2: Build context
context = build_llm_context(query, hits, grants)

# Step 2.5: Add SME knowledge if relevant
expert_knowledge = search_expert_knowledge(query, limit=3)

# Step 3: Call GPT
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": (
            build_user_prompt(query, grants)
            + "\n\nContext from grant documents:\n"
            + context
            + (f"\n\nExpert Knowledge from SME Database:\n{expert_knowledge}" if expert_knowledge.strip() else "")
        ),
    },
]
```

### 3. Integration in `/chat/stream` Endpoint

Location: [server.py:3000-3005](../src/api/server.py#L3000-L3005)

```python
user_content += f"RELEVANT GRANT CONTEXT:\n{context}\n\n"

# Add SME knowledge if relevant
expert_knowledge = search_expert_knowledge(query, limit=3)
if expert_knowledge.strip():
    user_content += f"EXPERT KNOWLEDGE FROM SME DATABASE:\n{expert_knowledge}\n\n"
```

## Database Structure

SME knowledge is stored in the `expert_examples` table (NOT `sme_knowledge`):

```sql
CREATE TABLE expert_examples (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    user_query TEXT NOT NULL,
    expert_response TEXT NOT NULL,
    client_context TEXT,
    grant_mentioned TEXT,
    notes TEXT,
    added_date TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    quality_score INTEGER DEFAULT 5
)
```

**TRL Example Entry**:
- ID: `expert_86e5fce3`
- Category: `general`
- Contains: Full UKRI TRL guidelines with URL
- URL: `https://www.ukri.org/councils/stfc/guidance-for-applicants/check-if-youre-eligible-for-funding/eligibility-of-technology-readiness-levels-trl/`

## Testing

Created test suite: [scripts/test_sme_retrieval.py](../scripts/test_sme_retrieval.py)

Run with:
```bash
python3 scripts/test_sme_retrieval.py
```

**Test Queries**:
- ✅ "is there a link to TRL levels"
- ✅ "TRL url"
- ✅ "where can I find TRL information"
- ✅ "what are TRL levels"
- ✅ "technology readiness levels link"

All tests pass and successfully retrieve the UKRI TRL URL.

## Search Algorithm

### Keyword Extraction
1. Split query into words
2. Convert to lowercase
3. Add domain-specific synonyms:
   - `trl` → `technology`, `readiness`, `level`
   - `link/url` → `http`, `www`, `resource`

### Scoring
- Base score: Count keyword matches in `user_query` + `expert_response`
- URL boost: +5 points when query contains 'link'/'url' and result contains URL
- Sort by score (descending)
- Return top N results

### Example Scoring

Query: "is there a link to TRL levels"

**Keywords extracted**: `['is', 'there', 'a', 'link', 'to', 'trl', 'levels']`

**Expanded keywords**: `['is', 'there', 'a', 'link', 'to', 'trl', 'levels', 'technology', 'readiness', 'level', 'http', 'www', 'resource']`

**Scoring for TRL entry**:
- Matches: 'trl' (3 times), 'technology' (2 times), 'readiness' (2 times), 'level' (2 times), 'http' (1 time)
- Base score: 10
- URL boost: +5 (because query has 'link' and result has 'http')
- **Total: 15**

## Benefits

1. **SME Knowledge Now Retrieved**: Questions like "is there a link to TRL levels" now retrieve the correct URL
2. **Context-Aware Responses**: LLM has access to both grant data AND expert knowledge
3. **Hybrid Search**: Combines keyword matching with relevance scoring
4. **Flexible**: Works with any SME knowledge ingested via Slack bot
5. **Low Overhead**: Only searches when relevant, returns top 3 results max

## Future Enhancements

- Add semantic/vector search for better fuzzy matching
- Track which SME knowledge is most useful (click tracking)
- Auto-tag expert examples by topic for faster filtering
- Add user feedback loop ("Was this helpful?")

---

**Status**: ✅ Implemented and Tested
**Date**: November 17, 2025
