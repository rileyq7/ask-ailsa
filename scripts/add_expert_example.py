#!/usr/bin/env python3
"""
Add expert examples to the database.
Can be used manually or imported by other scripts.
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "grants.db"

def add_expert_example(
    user_query: str,
    expert_response: str,
    category: str,
    client_context: str = None,
    grant_mentioned: str = None,
    notes: str = None,
    quality_score: int = 5
):
    """
    Add a new expert example to the database.

    Args:
        user_query: What the user asked
        expert_response: How the SME responded (can be full email or excerpt)
        category: Type of query (eligibility, strategy, grant_explanation, application_process, etc.)
        client_context: Brief description of client (optional)
        grant_mentioned: Which grant(s) discussed (optional)
        notes: Any additional context (optional)
        quality_score: 1-5 rating (5=excellent, use as example; 3=good; 1=reference only)
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    example_id = f"expert_{uuid.uuid4().hex[:8]}"
    added_date = datetime.now().isoformat()

    cur.execute("""
        INSERT INTO expert_examples
        (id, category, user_query, expert_response, client_context,
         grant_mentioned, notes, added_date, quality_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        example_id, category, user_query, expert_response,
        client_context, grant_mentioned, notes, added_date, quality_score
    ))

    conn.commit()
    conn.close()

    print(f"✅ Added expert example: {example_id}")
    print(f"   Category: {category}")
    print(f"   Grant: {grant_mentioned or 'N/A'}")
    print(f"   Quality: {quality_score}/5")
    return example_id


def add_from_email_file(email_file_path: str):
    """Parse an email file and interactively add as example."""
    with open(email_file_path, 'r') as f:
        email_content = f.read()

    print("\n" + "="*60)
    print("ADD EXPERT EXAMPLE FROM EMAIL")
    print("="*60)
    print(f"\nEmail preview:\n{email_content[:500]}...\n")

    # Interactive prompts
    print("What was the user's main question/context?")
    user_query = input("User query: ").strip()

    print("\nWhat category does this fall under?")
    print("Options: eligibility, strategy, grant_explanation, application_process, positioning, feasibility")
    category = input("Category: ").strip()

    print("\nWhich grant(s) are mentioned? (comma-separated if multiple)")
    grant_mentioned = input("Grant(s): ").strip() or None

    print("\nClient context (e.g., 'early-stage biotech', 'university spin-out')")
    client_context = input("Context: ").strip() or None

    print("\nQuality score (1-5, where 5=excellent example to show users)")
    quality_score = int(input("Score: ").strip() or "5")

    print("\nAny notes?")
    notes = input("Notes: ").strip() or None

    # Use full email as response
    expert_response = email_content

    # Confirm
    print("\n" + "="*60)
    print("CONFIRM:")
    print(f"  Query: {user_query}")
    print(f"  Category: {category}")
    print(f"  Grant: {grant_mentioned}")
    print(f"  Quality: {quality_score}/5")
    confirm = input("\nAdd this example? (y/n): ").strip().lower()

    if confirm == 'y':
        example_id = add_expert_example(
            user_query=user_query,
            expert_response=expert_response,
            category=category,
            client_context=client_context,
            grant_mentioned=grant_mentioned,
            notes=notes,
            quality_score=quality_score
        )
        print(f"\n✅ Successfully added example {example_id}")
    else:
        print("❌ Cancelled")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Load from file
        email_file = sys.argv[1]
        add_from_email_file(email_file)
    else:
        # Quick add mode
        print("Quick add mode (or use: python add_expert_example.py path/to/email.txt)")
        add_expert_example(
            user_query=input("User query: "),
            expert_response=input("Expert response: "),
            category=input("Category: "),
            quality_score=int(input("Quality (1-5): ") or "5")
        )
