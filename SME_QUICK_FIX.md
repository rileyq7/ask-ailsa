# SME Knowledge Not Working - Quick Fix Guide

## Problem
Chatbot says: "There isn't a universal set of documentation for TRL levels..."
But it SHOULD say: "Yes, UKRI provides comprehensive TRL guidance at https://www.ukri.org/..."

## Root Cause
According to your docs, the SME knowledge integration WAS implemented, but something's not working in production. Either:
1. The code isn't actually deployed
2. The database doesn't have the TRL entry
3. The integration is there but broken
4. API server needs restart

## Quick Diagnosis (2 minutes)

Run this diagnostic:
```bash
cd /Users/rileycoleman/grant-analyst-v2
python3 /mnt/user-data/outputs/debug_sme_not_working.py
```

This will tell you EXACTLY what's broken.

## Expected Issues & Fixes

### Issue 1: Database doesn't have TRL entry
**Diagnostic output**: `❌ ERROR: No TRL entries found in database!`

**Fix**:
```bash
# Add TRL entry manually
sqlite3 grants.db
```
```sql
INSERT INTO expert_examples (
    id, category, user_query, expert_response, 
    added_date, is_active, quality_score
) VALUES (
    'expert_trl_001',
    'general',
    'What is the official documentation for TRL levels in UK grant applications?',
    'UKRI (UK Research and Innovation) provides the official Technology Readiness Level (TRL) guidance. The TRL scale goes from 1 (basic principles observed) to 9 (actual system proven in operational environment). You can find the complete framework here: https://www.ukri.org/councils/stfc/guidance-for-applicants/check-if-youre-eligible-for-funding/eligibility-of-technology-readiness-levels-trl/',
    datetime('now'),
    1,
    5
);
```

Then restart API.

### Issue 2: search_expert_knowledge() not in server.py
**Diagnostic output**: `❌ Could not import search_expert_knowledge`

**Fix**: Add the function to server.py. Full code is in `/mnt/user-data/outputs/fix_sme_integration.py`

Copy the `search_expert_knowledge()` function and paste it into `src/api/server.py` around line 90.

### Issue 3: Chat endpoint doesn't call the function
**Diagnostic output**: `❌ stream_chat() does NOT call search_expert_knowledge()`

**Fix**: In `src/api/server.py`, find the `stream_chat()` function and add:

```python
# After building grant context
context = build_llm_context(query, hits, grants_for_llm)
user_content += f"RELEVANT GRANT CONTEXT:\n{context}\n\n"

# ADD THIS SECTION:
expert_knowledge = search_expert_knowledge(query, limit=3)
if expert_knowledge.strip():
    user_content += f"EXPERT KNOWLEDGE FROM SME DATABASE:\n{expert_knowledge}\n\n"
    logger.info(f"✓ Added {len(expert_knowledge)} chars of expert knowledge")

# Then continue with LLM call...
```

### Issue 4: API needs restart
**Diagnostic output**: All checks pass but chat still doesn't work

**Fix**:
```bash
# Kill and restart API
pkill -f "uvicorn.*server:app"
./start_api.sh
```

## Test After Fix

1. Restart API server
2. In Streamlit UI, ask: **"is there a link to TRL levels?"**
3. Response should now include: https://www.ukri.org/councils/stfc/...

## Quick Test via curl
```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "is there a link to TRL levels?",
    "conversation_history": []
  }'
```

Look for the UKRI URL in the streamed response.

## Verification Checklist

After applying fix:
- [ ] Ran diagnostic script
- [ ] Fixed identified issue
- [ ] Restarted API server
- [ ] Tested with TRL question
- [ ] Bot now provides UKRI link
- [ ] Checked API logs for "Added ... chars of expert knowledge"

## If Still Not Working

Check API logs while asking the question:
```bash
tail -f logs/api.log
# or wherever your logs are

# Look for:
# ✓ Added XXX chars of SME expert knowledge
```

If you see that log line, the integration is working but LLM is ignoring it. Update system prompt to emphasize expert knowledge more strongly.

If you DON'T see that log line, the function isn't being called - go back to Issue 3 fix.

## Files Provided

1. **debug_sme_not_working.py** - Comprehensive diagnostic
2. **fix_sme_integration.py** - Complete function code + integration instructions

Both in `/mnt/user-data/outputs/`

## Manual Test of search_expert_knowledge()

```python
# In Python interpreter:
import sys
sys.path.insert(0, '/Users/rileycoleman/grant-analyst-v2')
from src.api.server import search_expert_knowledge

result = search_expert_knowledge("is there a link to TRL levels", limit=3)
print(result)

# Should output text containing: https://www.ukri.org/councils/stfc/...
```

If this works but chat doesn't, the issue is integration (Issue 3).
If this fails, the issue is the function or database (Issues 1 or 2).
