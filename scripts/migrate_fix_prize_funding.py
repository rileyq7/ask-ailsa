#!/usr/bin/env python3
"""
Patch script: Fix prize-style funding amounts that weren't captured by standard parser.

Prize competitions often use different language patterns than standard grants:
- "share of a £1 million prize pot" instead of "up to £X million"
- "£250k each" or "£X per winner"
- "prize fund" instead of "total funding"

This script applies verified manual corrections for known prize competitions
and logs them in the audit trail with correction_type='manual_prize_patch'.

Usage:
    # Apply prize funding patches
    python3 scripts/migrate_fix_prize_funding.py --db grants.db

    # View audit log
    python3 scripts/migrate_fix_prize_funding.py --db grants.db --show-log
"""

import sqlite3
import argparse
from datetime import datetime


# ✅ Verified prize funding corrections (confirmed from UKRI/IUK Business Connect sources)
PRIZE_PATCHES = {
    "innovate_uk_2316": {
        "display": "share of a £1 million prize pot (3×£250k + £250k overall winner)",
        "gbp": 1_000_000,
        "reason": "Prize-style funding detected from UKRI/IUK Business Connect official sources",
        "source_urls": [
            "https://www.ukri.org/opportunity/expression-of-interest-the-agentic-ai-pioneers-prize/",
            "https://iuk-business-connect.org.uk/opportunities/the-agentic-ai-pioneers-prize/"
        ],
        "title": "The Agentic AI Pioneers Prize"
    },
    # Add more prize patches here as they are discovered
}


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


def log_correction(cur, grant_id, old_display, old_value, new_display, new_value, reason):
    """
    Record prize funding correction in audit table.

    Args:
        cur: Database cursor
        grant_id: Grant identifier
        old_display: Original funding display string (or NULL)
        old_value: Original numeric value (or NULL)
        new_display: Corrected display string
        new_value: Corrected numeric value
        reason: Explanation of correction
    """
    cur.execute("""
        INSERT INTO manual_corrections (
            id, old_total_fund, old_total_fund_gbp,
            new_total_fund, new_total_fund_gbp,
            reason, applied_at, correction_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        grant_id, old_display, old_value,
        new_display, new_value,
        reason, datetime.utcnow().isoformat(), 'manual_prize_patch'
    ))


def apply_prize_patches(cur):
    """
    Apply verified prize funding corrections.

    These corrections have been manually confirmed from official UKRI/IUK sources
    where prize competitions use non-standard funding language.

    Returns:
        int: Number of corrections applied
    """
    print("🏆 Applying verified prize funding patches...\n")

    applied = 0

    for grant_id, patch in PRIZE_PATCHES.items():
        cur.execute(
            "SELECT total_fund, total_fund_gbp FROM grants WHERE id = ?",
            (grant_id,)
        )
        row = cur.fetchone()

        if not row:
            print(f"⚠️  Skipped {grant_id}: not found in database")
            continue

        old_display, old_value = row
        new_display = patch["display"]
        new_value = patch["gbp"]

        # Skip if already patched
        if old_value == new_value and old_display == new_display:
            print(f"ℹ️  {grant_id}: already patched (£{new_value:,})")
            continue

        # Apply correction
        cur.execute("""
            UPDATE grants
            SET total_fund = ?, total_fund_gbp = ?
            WHERE id = ?
        """, (new_display, new_value, grant_id))

        # Build detailed reason with source URLs
        reason_detail = f"{patch['reason']} | Sources: {', '.join(patch['source_urls'])}"

        # Log correction
        log_correction(
            cur,
            grant_id,
            old_display,
            old_value,
            new_display,
            new_value,
            reason_detail
        )

        print(f"✅ {grant_id}: {patch['title']}")
        print(f"   Before: {old_display or '(no funding)'} ({old_value or 0:,})")
        print(f"   After:  {new_display} (£{new_value:,})")
        print(f"   Sources: {len(patch['source_urls'])} verified")
        print()

        applied += 1

    return applied


def run_prize_funding_patch(db_path="grants.db"):
    """
    Main function for prize funding patches - can be called from other scripts.

    Args:
        db_path: Path to SQLite database

    Returns:
        dict: Statistics about patches applied
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Ensure audit table exists
    ensure_audit_table(cur)

    # Apply patches
    applied = apply_prize_patches(cur)

    conn.commit()
    conn.close()

    return {
        "prize_patches_applied": applied,
        "total_prize_patches": len(PRIZE_PATCHES)
    }


def show_audit_log(db_path):
    """Display audit log of prize funding patches."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT
                id, old_total_fund, new_total_fund,
                reason, applied_at
            FROM manual_corrections
            WHERE correction_type = 'manual_prize_patch'
            ORDER BY applied_at DESC
            LIMIT 20
        """)

        rows = cur.fetchall()

        if not rows:
            print("📋 No prize funding patches found in audit log\n")
            return

        print(f"📋 Prize Funding Patches Audit Log (showing last {len(rows)} entries):\n")
        print("=" * 80)

        for row in rows:
            grant_id, old_val, new_val, reason, timestamp = row
            print(f"\n🏆 {grant_id}")
            print(f"   Before: {old_val or '(no funding)'}")
            print(f"   After:  {new_val}")
            print(f"   Reason: {reason}")
            print(f"   When:   {timestamp}")

        print("\n" + "=" * 80 + "\n")

    except sqlite3.OperationalError:
        print("📋 No audit log table found\n")

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Fix prize-style funding amounts not captured by standard parser"
    )
    parser.add_argument(
        "--db",
        default="grants.db",
        help="Path to SQLite database (default: grants.db)"
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Display audit log of prize patches instead of applying corrections"
    )

    args = parser.parse_args()

    if args.show_log:
        show_audit_log(args.db)
    else:
        print("=" * 80)
        print("🏆 Prize Funding Patch Migration Started")
        print("=" * 80)
        print(f"Database: {args.db}")
        print(f"Started:  {datetime.utcnow().isoformat()}")
        print("=" * 80)
        print()

        stats = run_prize_funding_patch(args.db)

        print("=" * 80)
        print("✅ Prize Funding Patch Complete")
        print("=" * 80)
        print(f"Patches applied: {stats['prize_patches_applied']}/{stats['total_prize_patches']}")
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()
