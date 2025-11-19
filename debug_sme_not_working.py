#!/usr/bin/env python3
"""
Debug why SME knowledge (TRL link) isn't appearing in chat responses.
"""

import sys
import sqlite3
sys.path.insert(0, '/Users/rileycoleman/grant-analyst-v2')

def check_sme_database():
    """Check if TRL data exists in expert_examples table."""
    print("=" * 80)
    print("1. CHECKING DATABASE FOR TRL KNOWLEDGE")
    print("=" * 80)
    
    try:
        conn = sqlite3.connect('/Users/rileycoleman/grant-analyst-v2/grants.db')
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='expert_examples'")
        if not cursor.fetchone():
            print("❌ ERROR: expert_examples table does NOT exist!")
            return False
        
        print("✅ expert_examples table exists\n")
        
        # Search for TRL entries
        cursor.execute("""
            SELECT id, category, user_query, expert_response, quality_score 
            FROM expert_examples 
            WHERE user_query LIKE '%TRL%' OR expert_response LIKE '%TRL%'
            OR user_query LIKE '%technology readiness%' OR expert_response LIKE '%technology readiness%'
        """)
        
        rows = cursor.fetchall()
        
        if not rows:
            print("❌ ERROR: No TRL entries found in database!")
            print("   The SME knowledge about TRL hasn't been ingested.")
            return False
        
        print(f"✅ Found {len(rows)} TRL-related entries:\n")
        
        for row in rows:
            entry_id, category, query, response, score = row
            print(f"   ID: {entry_id}")
            print(f"   Category: {category}")
            print(f"   Query: {query[:80]}...")
            print(f"   Response: {response[:100]}...")
            
            # Check if UKRI link is present
            if 'ukri.org' in response.lower():
                print("   ✅ Contains UKRI URL")
            else:
                print("   ⚠️  No UKRI URL found")
            print()
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


def test_search_function():
    """Test if search_expert_knowledge() function works."""
    print("=" * 80)
    print("2. TESTING search_expert_knowledge() FUNCTION")
    print("=" * 80)
    
    try:
        from src.api.server import search_expert_knowledge
        
        test_queries = [
            "is there a link to TRL levels",
            "TRL documentation",
            "technology readiness levels",
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 60)
            
            result = search_expert_knowledge(query, limit=3)
            
            if result and result.strip():
                print(f"✅ Returned {len(result)} characters")
                
                if 'ukri.org' in result.lower():
                    print("✅ Contains UKRI URL")
                    print(f"\nPreview:\n{result[:300]}...")
                else:
                    print("⚠️  No UKRI URL in result")
                    print(f"\nPreview:\n{result[:300]}...")
            else:
                print("❌ Returned empty/None")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import search_expert_knowledge: {e}")
        print("   The function may not exist in server.py")
        return False
    except Exception as e:
        print(f"❌ Error testing function: {e}")
        return False


def check_chat_integration():
    """Check if /chat/stream endpoint calls search_expert_knowledge."""
    print("\n" + "=" * 80)
    print("3. CHECKING CHAT ENDPOINT INTEGRATION")
    print("=" * 80)
    
    try:
        import inspect
        from src.api import server
        
        # Get the stream_chat function source
        if hasattr(server, 'stream_chat'):
            source = inspect.getsource(server.stream_chat)
            
            if 'search_expert_knowledge' in source:
                print("✅ stream_chat() calls search_expert_knowledge()")
                
                # Count how many times it's called
                count = source.count('search_expert_knowledge')
                print(f"   Called {count} time(s)")
                
                # Check if it's actually used in the prompt
                if 'expert_knowledge' in source.lower() or 'sme' in source.lower():
                    print("✅ Expert knowledge appears to be added to prompt")
                else:
                    print("⚠️  search_expert_knowledge is called but results may not be used")
                
                return True
            else:
                print("❌ stream_chat() does NOT call search_expert_knowledge()")
                print("   This is the problem - SME knowledge isn't being retrieved!")
                return False
        else:
            print("❌ stream_chat() function not found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking integration: {e}")
        return False


def check_chat_endpoint():
    """Check the non-streaming /chat endpoint too."""
    print("\n" + "=" * 80)
    print("4. CHECKING /chat ENDPOINT (non-streaming)")
    print("=" * 80)
    
    try:
        import inspect
        from src.api import server
        
        # Look for the chat function (non-streaming)
        # It might be called 'chat' or 'chat_completions' or similar
        
        # Get all functions in server module
        functions = [name for name, obj in inspect.getmembers(server) 
                    if inspect.isfunction(obj) and 'chat' in name.lower()]
        
        print(f"Found chat-related functions: {functions}\n")
        
        for func_name in functions:
            func = getattr(server, func_name)
            source = inspect.getsource(func)
            
            if 'search_expert_knowledge' in source:
                print(f"✅ {func_name}() calls search_expert_knowledge()")
            else:
                print(f"⚠️  {func_name}() does NOT call search_expert_knowledge()")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all diagnostic checks."""
    print("\n🔍 DIAGNOSING WHY SME KNOWLEDGE ISN'T APPEARING IN CHAT\n")
    
    results = []
    
    # Check 1: Database
    results.append(("Database has TRL data", check_sme_database()))
    
    # Check 2: Search function
    results.append(("search_expert_knowledge() works", test_search_function()))
    
    # Check 3: Integration
    results.append(("Chat endpoint calls function", check_chat_integration()))
    
    # Check 4: Other endpoints
    results.append(("Other endpoints checked", check_chat_endpoint()))
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    for check, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    
    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if not results[0][1]:
        print("\n❌ PRIMARY ISSUE: TRL data not in database")
        print("   SOLUTION: Run Slack bot to ingest SME knowledge, or manually add TRL entry")
        
    elif not results[1][1]:
        print("\n❌ PRIMARY ISSUE: search_expert_knowledge() function broken")
        print("   SOLUTION: Check server.py for function implementation errors")
        
    elif not results[2][1]:
        print("\n❌ PRIMARY ISSUE: Chat endpoint not calling search_expert_knowledge()")
        print("   SOLUTION: Add search_expert_knowledge() call to stream_chat() function")
        print("\n   Add this code in stream_chat():")
        print("   ```python")
        print("   # After building grant context, before calling LLM")
        print("   expert_knowledge = search_expert_knowledge(query, limit=3)")
        print("   if expert_knowledge.strip():")
        print("       user_content += f'\\n\\nEXPERT KNOWLEDGE:\\n{expert_knowledge}'")
        print("   ```")
        
    else:
        print("\n⚠️  UNCLEAR: All checks passed but chat still not working")
        print("   Possible issues:")
        print("   - search_expert_knowledge() returns empty for TRL queries")
        print("   - LLM is ignoring the expert knowledge in prompt")
        print("   - Caching issue (restart API server)")
        print("\n   SOLUTION: Check API logs when asking TRL question")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
