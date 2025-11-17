#!/usr/bin/env python3
"""
Migration script: Clean 'Funding competition' prefixes from grant titles.

This is a cosmetic cleanup for upstream formatting where Innovate UK includes
"Funding competition\n..." prefix in some competition titles. The prefix adds
no value and makes titles less readable in UI displays.

Features:
- Removes "Funding competition" prefix (with various separators: \n, : , -)
- Preserves original titles in audit log
- Logs all changes with correction_type='auto_title_cleanup'
- Idempotent (safe to run multiple times)

Usage:
    python3 scripts/migrate_clean_titles.py --db grants.db
    python3 scripts/migrate_clean_titles.py --db grants.db --show-log
    python3 scripts/migrate_clean_titles.py --db grants.db --dry-run
"""

import sqlite3
import argparse
import re
from datetime import datetime


# Pattern matches various formats of "Funding competition" prefix
# Examples:
#   "Funding competition\nActual Title"
#   "Funding competition: Actual Title"
#   "Funding competition - Actual Title"
#   "Funding Competition\nActual Title" (case insensitive)
TITLE_PREFIX_PATTERN = re.compile(
    r"^Funding\s+competition\s*[\n:;\-–—]\s*",
    re.IGNORECASE | re.MULTILINE
)


def ensure_audit_table(cur):
    """Create or ensure audit table exists for logging corrections."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS manual_corrections (
            id TEXT,
            old_total_fund TEXT,
            old_total_fund_gbp REAL,
            new_total_fund TEXT,
            new_total_fund_gbp REAL,
            reason TEXT,
            applied_at TEXT,
            correction_type TEXT
        )
    """)


def clean_title(title: str) -> str:
    """
    Remove 'Funding competition' prefix from title.

    Args:
        title: Original title string

    Returns:
        str: Cleaned title without prefix

    Examples:
        "Funding competition\\nQuantum Tech" -> "Quantum Tech"
        "Funding competition: AI Challenge" -> "AI Challenge"
        "Clean Title" -> "Clean Title" (unchanged)
    """
    if not title:
        return title

    cleaned = TITLE_PREFIX_PATTERN.sub("", title)

    # Strip any remaining leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


def find_titles_to_clean(cur):
    """
    Find all grants with 'Funding competition' prefix in title.

    Args:
        cur: Database cursor

    Returns:
        list: List of (id, current_title) tuples needing cleanup
    """
    cur.execute("""
        SELECT id, title
        FROM grants
        WHERE title LIKE 'Funding competition%'
           OR title LIKE 'funding competition%'
        ORDER BY id
    """)

    return cur.fetchall()


def apply_title_cleanup(db_path, dry_run=False):
    """
    Clean 'Funding competition' prefixes from grant titles.

    Args:
        db_path: Path to SQLite database
        dry_run: If True, show what would be changed without modifying database

    Returns:
        dict: Statistics about titles cleaned
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if not dry_run:
        ensure_audit_table(cur)

    # Find titles to clean
    candidates = find_titles_to_clean(cur)

    if not candidates:
        print("✅ No titles with 'Funding competition' prefix found\n")
        conn.close()
        return {
            "cleaned": 0,
            "skipped": 0,
            "total": 0
        }

    print(f"🧹 Found {len(candidates)} title(s) to clean...\n")

    cleaned = 0
    skipped = 0

    for grant_id, old_title in candidates:
        new_title = clean_title(old_title)

        # Skip if no change (shouldn't happen given SQL filter, but safe)
        if old_title == new_title:
            print(f"ℹ️  {grant_id}: no change needed")
            skipped += 1
            continue

        if dry_run:
            print(f"🔍 {grant_id} [DRY RUN]")
            print(f"   Before: {old_title[:70]}...")
            print(f"   After:  {new_title[:70]}...")
            print()
            cleaned += 1
        else:
            # Apply change
            cur.execute("""
                UPDATE grants
                SET title = ?
                WHERE id = ?
            """, (new_title, grant_id))

            # Log to audit table
            # Note: old_total_fund and new_total_fund are NULL since this is title cleanup
            cur.execute("""
                INSERT INTO manual_corrections (
                    id, old_total_fund, old_total_fund_gbp,
                    new_total_fund, new_total_fund_gbp,
                    reason, applied_at, correction_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                grant_id,
                old_title,  # Store old title in old_total_fund field
                None,
                new_title,  # Store new title in new_total_fund field
                None,
                "Removed 'Funding competition' prefix for cleaner display",
                datetime.utcnow().isoformat(),
                "auto_title_cleanup"
            ))

            print(f"✅ {grant_id}")
            print(f"   Before: {old_title[:70]}...")
            print(f"   After:  {new_title[:70]}...")
            print()

            cleaned += 1

    if not dry_run:
        conn.commit()

    conn.close()

    return {
        "cleaned": cleaned,
        "skipped": skipped,
        "total": len(candidates)
    }


def show_audit_log(db_path):
    """Display audit log of title cleanups."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT
                id, old_total_fund, new_total_fund, applied_at
            FROM manual_corrections
            WHERE correction_type = 'auto_title_cleanup'
            ORDER BY applied_at DESC
            LIMIT 20
        """)

        rows = cur.fetchall()

        if not rows:
            print("📋 No title cleanups found in audit log\n")
            return

        print(f"📋 Title Cleanup Audit Log (showing last {len(rows)} entries):\n")
        print("=" * 80)

        for row in rows:
            grant_id, old_title, new_title, timestamp = row
            print(f"\n🧹 {grant_id}")
            print(f"   Before: {old_title[:70]}...")
            print(f"   After:  {new_title[:70]}...")
            print(f"   When:   {timestamp}")

        print("\n" + "=" * 80 + "\n")

    except sqlite3.OperationalError:
        print("📋 No audit log table found\n")

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Clean 'Funding competition' prefixes from grant titles"
    )
    parser.add_argument(
        "--db",
        default="grants.db",
        help="Path to SQLite database (default: grants.db)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying database"
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Display audit log instead of applying cleanup"
    )

    args = parser.parse_args()

    if args.show_log:
        show_audit_log(args.db)
    else:
        mode = "[DRY RUN]" if args.dry_run else ""

        print("=" * 80)
        print(f"🔧 Title Cleanup Migration {mode}")
        print("=" * 80)
        print(f"Database: {args.db}")
        print(f"Started:  {datetime.utcnow().isoformat()}")
        if args.dry_run:
            print("Mode:     DRY RUN (no changes will be made)")
        print("=" * 80)
        print()

        stats = apply_title_cleanup(args.db, dry_run=args.dry_run)

        print("=" * 80)
        print(f"✅ Title Cleanup {'Simulation' if args.dry_run else 'Complete'}")
        print("=" * 80)
        print(f"Cleaned:  {stats['cleaned']} title(s)")
        print(f"Skipped:  {stats['skipped']} title(s)")
        print(f"Total:    {stats['total']} grant(s) checked")
        if args.dry_run:
            print()
            print("Run without --dry-run to apply changes.")
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()
