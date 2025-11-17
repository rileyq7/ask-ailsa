#!/usr/bin/env python3
"""
View all expert examples in the database.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "grants.db"

def view_all_examples():
    """View all expert examples."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, category, user_query, grant_mentioned, quality_score, added_date
        FROM expert_examples
        WHERE is_active = 1
        ORDER BY quality_score DESC, added_date DESC
    """)

    print("\n" + "="*80)
    print("EXPERT EXAMPLES LIBRARY")
    print("="*80)

    rows = cur.fetchall()

    if not rows:
        print("\n⚠️  No examples found. Add some using:")
        print("  python scripts/add_expert_example.py")
        print("  python scripts/import_one_pager.py one_pagers/")
        conn.close()
        return

    for row in rows:
        print(f"\nID: {row[0]}")
        print(f"Category: {row[1]}")
        print(f"Query: {row[2][:80]}...")
        print(f"Grant: {row[3] or 'N/A'}")
        print(f"Quality: {'⭐' * row[4]} ({row[4]}/5)")
        print(f"Added: {row[5][:10]}")
        print("-" * 80)

    print(f"\n📊 Total: {len(rows)} examples")

    conn.close()

if __name__ == "__main__":
    view_all_examples()
