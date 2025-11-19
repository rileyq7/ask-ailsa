# SME Knowledge Architecture

## Purpose

The SME knowledge system stores two distinct types of information:

1. **Factual Knowledge** - URLs, documentation, guidance that should be shared with users
2. **Tone Examples** - Internal Ailsa curations showing communication style (NEVER shared)

## Data Separation

### AILSA INTERNAL Content (Tone Reference Only)

**Markers:** Any entry with "AILSA INTERNAL" in:
- `user_query`
- `expert_response`
- `grant_mentioned`

**Purpose:** Train the LLM on Ailsa's voice, tone, phrasing, and approach

**Usage:**
- ✅ Include in system prompts as style examples
- ✅ Use to teach natural, conversational tone
- ✅ Reference for how to structure recommendations
- ❌ NEVER return as factual information to users
- ❌ NEVER use content from these as answers

**Function:** `get_ailsa_tone_examples(limit=2)`

### Public Knowledge (Factual Information)

**Markers:** All other entries (NO "AILSA INTERNAL" marker)

**Purpose:** Provide factual information to users (URLs, TRL documentation, etc.)

**Usage:**
- ✅ Return to users when relevant
- ✅ Include in chat responses
- ✅ Use as source of truth for questions

**Function:** `search_expert_knowledge(query, limit=5)`

## Implementation

### Filtering in `search_expert_knowledge()`

```python
# SQL filters out AILSA INTERNAL
cursor.execute("""
    SELECT user_query, expert_response, category, grant_mentioned
    FROM expert_examples
    WHERE is_active = 1
      AND user_query NOT LIKE '%AILSA INTERNAL%'
      AND expert_response NOT LIKE '%AILSA INTERNAL%'
      AND grant_mentioned NOT LIKE '%AILSA INTERNAL%'
    ...
""")

# Double-check in Python
if 'AILSA INTERNAL' in user_query or 'AILSA INTERNAL' in expert_response:
    continue  # Skip this entry
```

### Retrieving Tone Examples

```python
def get_ailsa_tone_examples(limit=2):
    """Get INTERNAL examples for tone reference ONLY"""
    cursor.execute("""
        SELECT expert_response
        FROM expert_examples
        WHERE (user_query LIKE '%AILSA INTERNAL%'
               OR expert_response LIKE '%AILSA INTERNAL%'
               OR grant_mentioned LIKE '%AILSA INTERNAL%')
        ...
    """)
```

## Example Data

### Factual Knowledge Entry
```
user_query: "What is the official documentation for TRL levels?"
expert_response: "UKRI provides TRL guidance at https://www.ukri.org/..."
grant_mentioned: NULL
```
→ **Returned to users** when they ask about TRL

### Tone Example Entry
```
user_query: "Tell me about AILSA INTERNAL USE ONLY"
expert_response: "AILSA INTERNAL USE ONLY\n\nOpportunity / Funding Stream...\n\nFood for thought: We think there may be..."
grant_mentioned: "AILSA INTERNAL USE ONLY"
```
→ **Used for tone training** but NEVER returned to users

## Adding New Content

### Adding Factual Knowledge
```python
# Via Slack #sme-knowledge channel or scripts/sme/add_expert_example.py
# Do NOT include "AILSA INTERNAL" anywhere
{
    "user_query": "Where can I find X?",
    "expert_response": "You can find X at https://...",
    "grant_mentioned": None
}
```

### Adding Tone Examples
```python
# Import one-pagers with scripts/sme/import_one_pager.py
# These automatically get marked as AILSA INTERNAL
{
    "user_query": "Tell me about AILSA INTERNAL USE ONLY",
    "expert_response": "AILSA INTERNAL USE ONLY\n\n[curated content]",
    "grant_mentioned": "AILSA INTERNAL USE ONLY"
}
```

## Verification

Test that filtering works:

```python
from src.api.server import search_expert_knowledge, get_ailsa_tone_examples

# Should NOT include AILSA INTERNAL
result = search_expert_knowledge("networking opportunities")
assert "AILSA INTERNAL" not in result

# SHOULD include AILSA INTERNAL
tone = get_ailsa_tone_examples(limit=2)
assert "AILSA INTERNAL" in tone or len(tone) > 0
```

## Why This Matters

**Risk if internal content leaks:**
- Users see "AILSA INTERNAL USE ONLY" in responses (unprofessional)
- Client-specific advice gets shared with wrong companies
- One-pager content intended for learning gets presented as fact
- Confusing mix of style examples and actual information

**Proper separation ensures:**
- Users only see relevant, factual information
- LLM learns Ailsa's voice without exposing internal content
- Clean separation of training data vs. production data
- Professional, polished user experience

---

**Last Updated:** November 18, 2025  
**Related Files:**
- `src/api/server.py` - Functions: `search_expert_knowledge()`, `get_ailsa_tone_examples()`
- `scripts/sme/import_one_pager.py` - Adds AILSA INTERNAL tone examples
- `scripts/sme/add_expert_example.py` - Adds public factual knowledge
