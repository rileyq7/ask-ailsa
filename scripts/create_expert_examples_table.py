#!/usr/bin/env python3
"""
Create database table for storing expert response examples.
Run once to set up the schema.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "grants.db"

def create_expert_examples_table():
    """Create table for storing expert response examples."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS expert_examples (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,  -- e.g., 'eligibility', 'strategy', 'grant_explanation'
            user_query TEXT NOT NULL,
            expert_response TEXT NOT NULL,
            client_context TEXT,  -- e.g., 'early-stage biotech', 'university spin-out'
            grant_mentioned TEXT,  -- e.g., 'Biomedical Catalyst', 'Innovation Loans'
            notes TEXT,  -- Any additional context
            added_date TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,  -- Can disable without deleting
            quality_score INTEGER DEFAULT 5  -- 1-5 rating for filtering best examples
        )
    """)

    # Create indexes for faster lookups
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_expert_category
        ON expert_examples(category)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_expert_grant
        ON expert_examples(grant_mentioned)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_expert_quality
        ON expert_examples(quality_score)
    """)

    conn.commit()
    conn.close()
    print("✅ Created expert_examples table")

if __name__ == "__main__":
    create_expert_examples_table()
