# Grant Recommendation Filtering Fix

## Problem

The bot was recommending grant cards on **every single message**, even when inappropriate:
- "What are SWATs?" → Shows grants (inappropriate)
- "Do you recommend a grant on every message?" → Shows grants (ironic!)
- "Thanks!" → Shows grants (unnecessary)

This made the bot feel pushy and sales-y instead of helpful.

## Solution

Added smart filtering function `should_include_grant_recommendations()` that only shows grant cards when genuinely relevant.

## When Grants Are NOT Shown

### 1. Meta Questions About the Bot
```
❌ "Do you recommend a grant on every message?"
❌ "Why do you always show grants?"
❌ "How do you work?"
❌ "What do you do?"
```

### 2. General Definition/Explanation Questions
```
❌ "What are SWATs?"
❌ "What does TRL mean?"
❌ "Explain clinical trials"
❌ "What's a consortium?"
❌ "Define match funding"
```

**Exception:** Grant-related definitions DO show grants:
```
✅ "What's the NIHR i4i grant?"  → Shows grants
```

### 3. Simple Acknowledgments
```
❌ "Thanks!"
❌ "Got it"
❌ "OK"
❌ "Great"
```

### 4. Follow-up Clarifications (short)
```
❌ "Why?"
❌ "How come?"
❌ "What do you mean?"
```

## When Grants ARE Shown

### Explicit Grant Searches
```
✅ "Show me NIHR grants"
✅ "Find funding for AI"
✅ "What grants are available?"
```

### Grant-Specific Questions
```
✅ "When is the deadline for i4i?"
✅ "How much funding does Innovate UK offer?"
✅ "Who can apply for biomedical catalyst?"
```

### Funding Requests
```
✅ "I need funding for my startup"
✅ "Looking for clinical trial grants"
✅ "Help me find opportunities"
```

## Implementation

### Function Added
[src/api/server.py](../src/api/server.py):387-462

```python
def should_include_grant_recommendations(query: str, response: str) -> bool:
    """
    Determine if we should include grant card recommendations.
    NOT every query needs grant cards!
    """
    # Meta questions → NO
    # Definitions (unless grant-related) → NO
    # Acknowledgments → NO
    # Grant searches → YES
    # Returns: True if grants should be shown
```

### Endpoints Updated

#### 1. `/chat` endpoint (line 2134)
```python
should_show_grants = should_include_grant_recommendations(query, answer_markdown)

if should_show_grants:
    # Build grant cards
else:
    # Return empty grant list
```

#### 2. `/chat/stream` endpoint (line 3018)
```python
should_show_grants = should_include_grant_recommendations(query, full_response)

if should_show_grants and enriched_grant_refs:
    yield grants
else:
    # Skip grants
```

## Testing

Run the test suite:
```bash
python3 scripts/test_grant_filtering.py
```

**All 13 tests pass:**
- 7 tests for queries that should NOT show grants
- 6 tests for queries that should show grants

## Examples

### Before Fix
```
User: "What are SWATs?"
Bot: "SWATs are studies within a trial..."
     [3 Grant Cards Shown] ❌
```

### After Fix
```
User: "What are SWATs?"
Bot: "SWATs are studies within a trial..."
     [No Grant Cards] ✅
```

---

```
User: "Show me NIHR grants"
Bot: "Here are some NIHR opportunities..."
     [3 Grant Cards Shown] ✅
```

## Benefits

1. **More Natural Conversation** - Feels like a real advisor, not a salesperson
2. **Better UX** - Grant cards only when helpful
3. **Reduced Clutter** - Clean responses to definition questions
4. **No More Irony** - Meta questions answered without contradicting themselves

## Logging

The function logs its decisions:
```
INFO: Meta question detected - skipping grant cards
INFO: General definition question - skipping grant cards
INFO: Grant-seeking query detected - including grant cards
INFO: No grant intent detected - skipping grant cards
```

Check logs to see filtering in action!

## Future Enhancements

Potential improvements:
- Track conversation history to avoid re-showing same grants
- Add user preference for "always show" / "never show" grants
- Learn from user behavior (clicks) to refine filtering

---

**Status:** ✅ Implemented and Tested
**Date:** November 17, 2025
