"""
Fix to ensure SME knowledge is retrieved and included in chat responses.

PROBLEM: Chatbot doesn't provide UKRI TRL link even though it's in the database.

SOLUTION: Ensure search_expert_knowledge() is called in the chat streaming endpoint.
"""

# ============================================================================
# STEP 1: Verify search_expert_knowledge() function exists
# ============================================================================
# This function should be in server.py around line 92-164
# If it's NOT there, add it:

def search_expert_knowledge(query: str, limit: int = 3) -> str:
    """
    Search expert_examples table for relevant SME knowledge.
    
    Args:
        query: User's question
        limit: Max number of results to return
        
    Returns:
        Formatted string with expert knowledge, or empty string if none found
    """
    import sqlite3
    import re
    from pathlib import Path
    
    db_path = Path("grants.db")  # Adjust path if needed
    
    if not db_path.exists():
        return ""
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Extract keywords from query
        query_lower = query.lower()
        keywords = re.findall(r'\b\w+\b', query_lower)
        
        # Remove stopwords
        stopwords = {'is', 'are', 'the', 'a', 'an', 'to', 'for', 'of', 'in', 'on', 'at'}
        keywords = [k for k in keywords if k not in stopwords]
        
        # Add synonyms
        expanded_keywords = keywords.copy()
        synonym_map = {
            'trl': ['technology', 'readiness', 'level'],
            'link': ['url', 'http', 'www', 'resource'],
            'url': ['link', 'http', 'www', 'resource'],
        }
        
        for keyword in keywords:
            if keyword in synonym_map:
                expanded_keywords.extend(synonym_map[keyword])
        
        # Get all active expert examples
        cursor.execute("""
            SELECT user_query, expert_response, quality_score
            FROM expert_examples
            WHERE is_active = 1
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return ""
        
        # Score each example
        scored_examples = []
        for user_query, expert_response, quality_score in rows:
            combined_text = f"{user_query} {expert_response}".lower()
            
            # Count keyword matches
            score = sum(1 for kw in expanded_keywords if kw in combined_text)
            
            # Boost score if asking for link and response contains URL
            if any(term in query_lower for term in ['link', 'url', 'documentation', 'resource']):
                if any(indicator in combined_text for indicator in ['http', 'www', 'ukri.org']):
                    score += 5
            
            if score > 0:
                scored_examples.append((score, user_query, expert_response, quality_score))
        
        if not scored_examples:
            return ""
        
        # Sort by score
        scored_examples.sort(reverse=True)
        
        # Format top results
        results = []
        for score, user_query, expert_response, quality_score in scored_examples[:limit]:
            results.append(f"Q: {user_query}\nA: {expert_response}")
        
        return "\n\n".join(results)
        
    except Exception as e:
        import logging
        logging.error(f"Error searching expert knowledge: {e}")
        return ""


# ============================================================================
# STEP 2: Add to chat streaming endpoint
# ============================================================================
# Find the stream_chat() function in server.py
# Look for where it builds the user prompt, around the line that says:
#   user_content = f"User query: {query}\n\n"
#
# Then add this code AFTER building grant context but BEFORE calling the LLM:

"""
Example integration in stream_chat():

async def stream_chat(...):
    # ... existing code ...
    
    # Build grant context
    context = build_llm_context(query, hits, grants_for_llm)
    user_content += f"RELEVANT GRANT CONTEXT:\n{context}\n\n"
    
    # ⭐ ADD THIS SECTION:
    # Search SME knowledge
    expert_knowledge = search_expert_knowledge(query, limit=3)
    if expert_knowledge.strip():
        user_content += f"EXPERT KNOWLEDGE FROM SME DATABASE:\n{expert_knowledge}\n\n"
        logger.info(f"Added {len(expert_knowledge)} chars of expert knowledge to prompt")
    
    # Continue with LLM call...
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
"""


# ============================================================================
# STEP 3: Verify it works
# ============================================================================
# After adding the code above:
# 1. Restart your API server
# 2. Ask: "is there a link to TRL levels?"
# 3. Check logs for: "Added ... chars of expert knowledge to prompt"
# 4. Response should now include the UKRI URL


# ============================================================================
# EXACT CODE TO ADD
# ============================================================================
# Here's the exact code block to add to your stream_chat() function:

EXACT_CODE_TO_ADD = """
    # Add SME expert knowledge if relevant
    expert_knowledge = search_expert_knowledge(query, limit=3)
    if expert_knowledge.strip():
        user_content += f'''

EXPERT KNOWLEDGE FROM SME DATABASE:
{expert_knowledge}

'''
        logger.info(f"✓ Added {len(expert_knowledge)} chars of SME expert knowledge")
"""

# Location to add it: 
# In stream_chat(), find this section:
#
#   context = build_llm_context(query, hits, grants_for_llm)
#   user_content += f"RELEVANT GRANT CONTEXT:\n{context}\n\n"
#
# Add the EXACT_CODE_TO_ADD right after those lines, before the LLM call.


# ============================================================================
# ALTERNATIVE: Update system prompt
# ============================================================================
# You can also update SYSTEM_PROMPT to explicitly tell the LLM to use expert knowledge:

UPDATED_SYSTEM_PROMPT_SECTION = """
You have access to:
1. Grant information from the vector database
2. Expert knowledge from our SME curator database (contains links, guidance, best practices)

When answering:
- If expert knowledge contains relevant links or resources, ALWAYS include them in your response
- Cite expert knowledge when it provides useful context
- Prefer official documentation links from expert knowledge over general explanations
"""

print("=" * 80)
print("SME KNOWLEDGE FIX - IMPLEMENTATION GUIDE")
print("=" * 80)
print("\n1. Verify search_expert_knowledge() exists in server.py")
print("2. Add the function call to stream_chat() as shown above")
print("3. Restart API server")
print("4. Test with: 'is there a link to TRL levels?'")
print("5. Response should include: https://www.ukri.org/councils/stfc/...")
print("\n" + "=" * 80)
