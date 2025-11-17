#!/usr/bin/env python3
"""
Test script to verify SME knowledge retrieval for TRL links.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the search function
import importlib.util
spec = importlib.util.spec_from_file_location(
    "server",
    "/Users/rileycoleman/grant-analyst-v2/src/api/server.py"
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)

def test_trl_retrieval():
    """Test that TRL URL is retrieved when user asks for it."""
    print("🧪 Testing SME Knowledge Retrieval for TRL Links\n")
    print("="*60)

    test_queries = [
        "is there a link to TRL levels",
        "TRL url",
        "where can I find TRL information",
        "what are TRL levels",
        "technology readiness levels link",
    ]

    for query in test_queries:
        print(f"\n📋 Query: '{query}'")
        print("-"*60)

        result = server.search_expert_knowledge(query, limit=3)

        if result.strip():
            print("✅ Found SME knowledge:")
            print(result[:500])  # Show first 500 chars

            # Check if it contains the UKRI TRL URL
            if "ukri.org" in result.lower() or "http" in result.lower():
                print("\n✅ CONTAINS URL - Success!")
            else:
                print("\n⚠️  No URL found in result")
        else:
            print("❌ No SME knowledge found")

        print()

    print("="*60)
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_trl_retrieval()
