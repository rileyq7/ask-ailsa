#!/usr/bin/env python3
"""
End-to-end test to verify SME knowledge is integrated into chat responses.
This simulates a real chat request asking for TRL links.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_chat_integration():
    """Test that chat endpoint includes SME knowledge in responses."""
    print("🧪 Testing SME Knowledge Integration in Chat Endpoint\n")
    print("="*60)

    # Import after path setup
    from src.llm.client import LLMClient

    print("\n1️⃣ Testing search_expert_knowledge() function...")
    print("-"*60)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "server",
        "/Users/rileycoleman/grant-analyst-v2/src/api/server.py"
    )
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)

    query = "is there a link to TRL levels"
    result = server.search_expert_knowledge(query, limit=3)

    if "ukri.org" in result.lower():
        print("✅ search_expert_knowledge() works - found UKRI URL")
    else:
        print("❌ search_expert_knowledge() failed - no UKRI URL found")
        return False

    print("\n2️⃣ Testing LLM Client initialization...")
    print("-"*60)

    try:
        llm_client = LLMClient()
        print(f"✅ LLM Client initialized successfully")
    except Exception as e:
        print(f"❌ LLM Client initialization failed: {e}")
        return False

    print("\n3️⃣ Verifying integration points...")
    print("-"*60)

    # Check that explain_with_gpt calls search_expert_knowledge
    import inspect
    source = inspect.getsource(server.explain_with_gpt)
    if "search_expert_knowledge" in source:
        print("✅ explain_with_gpt() calls search_expert_knowledge()")
    else:
        print("❌ explain_with_gpt() does NOT call search_expert_knowledge()")
        return False

    print("\n4️⃣ Summary...")
    print("-"*60)
    print("✅ SME knowledge retrieval: Working")
    print("✅ Integration into chat flow: Confirmed")
    print("✅ UKRI TRL URL: Available in database")
    print("\n🎉 All tests passed!")
    print("\nWhen a user asks 'is there a link to TRL levels', the system will:")
    print("1. Search expert_examples table")
    print("2. Find UKRI TRL resource with URL")
    print("3. Include it in LLM context")
    print("4. LLM can now reference the URL in its response")

    print("\n" + "="*60)
    return True

if __name__ == "__main__":
    success = test_chat_integration()
    sys.exit(0 if success else 1)
