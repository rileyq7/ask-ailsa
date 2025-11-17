#!/usr/bin/env python3
"""
Debug script to check what SME knowledge exists and why it's not being retrieved.
"""

import sqlite3
import sys
import os

DB_PATH = "/Users/rileycoleman/grant-analyst-v2/grants.db"


def check_sme_table_exists():
    """Check if sme_knowledge table exists."""
    print("🔍 Checking for SME knowledge table...\n")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    print(f"Found {len(tables)} tables in database:")
    for table in tables:
        print(f"  • {table}")

    conn.close()

    if 'sme_knowledge' in tables:
        print("\n✅ sme_knowledge table EXISTS!")
        return True
    else:
        print("\n❌ sme_knowledge table DOES NOT EXIST")
        print("\nThis is why your TRL link isn't being found!")
        print("\nTo fix: Run the Slack bot setup:")
        print("  python slack/sme_slack_bot.py --setup")
        return False


def check_sme_content():
    """Check what's actually stored in SME knowledge."""
    print("\n" + "="*60)
    print("📚 Checking SME Knowledge Content")
    print("="*60 + "\n")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Count total entries
        cursor.execute("SELECT COUNT(*) FROM sme_knowledge")
        total = cursor.fetchone()[0]
        print(f"Total SME entries: {total}")

        if total == 0:
            print("\n⚠️  Database table exists but is EMPTY!")
            print("   The Slack bot hasn't ingested any knowledge yet.")
            conn.close()
            return

        # Check for TRL-related content
        print("\n🔍 Searching for TRL-related entries...")
        cursor.execute("""
            SELECT id, content, topics, source_metadata, created_at
            FROM sme_knowledge
            WHERE content LIKE '%trl%'
               OR content LIKE '%technology readiness%'
               OR content LIKE '%innovate%'
               OR topics LIKE '%trl%'
            ORDER BY created_at DESC
            LIMIT 5
        """)

        trl_results = cursor.fetchall()

        if trl_results:
            print(f"✅ Found {len(trl_results)} TRL-related entries:\n")
            for entry_id, content, topics, metadata, created in trl_results:
                print(f"Entry ID: {entry_id}")
                print(f"Created: {created}")
                print(f"Topics: {topics}")
                print(f"Source: {metadata}")
                print(f"Content preview: {content[:300]}...")
                print("-" * 60)
        else:
            print("❌ No TRL-related content found in SME knowledge!")
            print("   Your URL might not have been ingested correctly.")

        # Check for URLs in general
        print("\n🔗 Checking for URL resources...")
        cursor.execute("""
            SELECT COUNT(*) FROM sme_knowledge
            WHERE content LIKE '%http%' OR content LIKE '%www.%'
        """)
        url_count = cursor.fetchone()[0]
        print(f"Entries containing URLs: {url_count}")

        if url_count > 0:
            cursor.execute("""
                SELECT id, content
                FROM sme_knowledge
                WHERE content LIKE '%http%' OR content LIKE '%www.%'
                ORDER BY created_at DESC
                LIMIT 3
            """)

            print("\nRecent URL entries:")
            for entry_id, content in cursor.fetchall():
                # Extract URLs
                import re
                urls = re.findall(r'https?://[^\s]+', content)
                print(f"  • Entry {entry_id}: {urls[0] if urls else 'No URL found'}")

    except sqlite3.OperationalError as e:
        print(f"\n❌ Error: {e}")
        print("   The sme_knowledge table might have a different schema.")

    conn.close()


def test_search_query():
    """Test how a typical TRL query would search SME knowledge."""
    print("\n" + "="*60)
    print("🧪 Testing Search for: 'is there a link to TRL levels'")
    print("="*60 + "\n")

    query = "is there a link to TRL levels"
    keywords = ['trl', 'link', 'level', 'technology', 'readiness']

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Simulate keyword search
        print("Searching with keywords: " + ", ".join(keywords))

        conditions = ' OR '.join([f"content LIKE '%{kw}%'" for kw in keywords])

        cursor.execute(f"""
            SELECT id, content
            FROM sme_knowledge
            WHERE {conditions}
            LIMIT 5
        """)

        results = cursor.fetchall()

        if results:
            print(f"\n✅ Found {len(results)} matching entries!")
            for entry_id, content in results:
                print(f"\nEntry {entry_id}:")
                print(f"{content[:400]}...")
        else:
            print("\n❌ No results found with keyword search")
            print("   This confirms why your query isn't finding the TRL link!")

    except sqlite3.OperationalError as e:
        print(f"❌ Error: {e}")

    conn.close()


def main():
    """Run all diagnostic checks."""
    print("\n" + "🔧"*30)
    print("SME Knowledge Diagnostic Tool")
    print("🔧"*30 + "\n")

    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        print("\nMake sure you're in the right directory!")
        sys.exit(1)

    # Check for table
    table_exists = check_sme_table_exists()

    if table_exists:
        # Check content
        check_sme_content()

        # Test search
        test_search_query()

    print("\n" + "="*60)
    print("📋 Summary & Next Steps")
    print("="*60 + "\n")

    if not table_exists:
        print("1. Create SME knowledge table:")
        print("   python slack/sme_slack_bot.py --setup\n")
        print("2. Start the Slack bot to ingest knowledge:")
        print("   cd slack && ./start_slack_bot.sh\n")
        print("3. Add your TRL URL in the #sme-knowledge channel")
    else:
        print("✅ SME knowledge system is set up")
        print("\nTo integrate with chat:")
        print("  1. Add SME retrieval to /chat endpoint")
        print("  2. Use hybrid search (vector + keyword)")
        print("  3. Test with TRL queries")

    print("\n" + "🔧"*30 + "\n")


if __name__ == "__main__":
    main()
