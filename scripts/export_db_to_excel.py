#!/usr/bin/env python3
"""
Export grants database to Excel workbook for SME review.

Creates a multi-sheet Excel file with:
- Grants: Core grant information (titles, funding, dates)
- Documents: Document coverage per grant
- Corrections: Audit log of all data corrections
- Statistics: Summary metrics and data quality stats

Usage:
    python3 scripts/export_db_to_excel.py --db grants.db --out grants_review.xlsx
    python3 scripts/export_db_to_excel.py --db grants.db --out grants_review.xlsx --include-embeddings
"""

import sqlite3
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path


def export_grants_sheet(conn):
    """
    Export main grants data.

    Returns:
        pd.DataFrame: Grant information suitable for SME review
    """
    query = """
    SELECT
        id AS grant_id,
        title,
        source,
        total_fund AS funding_display,
        total_fund_gbp AS funding_gbp,
        CASE
            WHEN total_fund_gbp IS NULL THEN 'N/A'
            WHEN total_fund_gbp >= 1000000 THEN
                '£' || ROUND(total_fund_gbp / 1000000.0, 2) || 'M'
            WHEN total_fund_gbp >= 1000 THEN
                '£' || ROUND(total_fund_gbp / 1000.0, 0) || 'K'
            ELSE '£' || total_fund_gbp
        END AS funding_formatted,
        CASE
            WHEN is_active = 1 THEN 'active'
            WHEN datetime(opens_at) > datetime('now') THEN 'upcoming'
            ELSE 'closed'
        END AS status,
        is_active,
        opens_at,
        closes_at,
        url AS source_url,
        created_at,
        updated_at
    FROM grants
    ORDER BY
        CASE
            WHEN is_active = 1 THEN 1
            WHEN datetime(opens_at) > datetime('now') THEN 2
            ELSE 3
        END,
        closes_at DESC
    """

    df = pd.read_sql_query(query, conn)

    # Format dates for readability
    date_cols = ['opens_at', 'closes_at', 'created_at', 'updated_at']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')

    # Convert is_active to readable text
    df['is_active'] = df['is_active'].map({1: 'Yes', 0: 'No', None: 'Unknown'})

    return df


def export_documents_sheet(conn):
    """
    Export document coverage summary.

    Returns:
        pd.DataFrame: Document counts and types per grant
    """
    query = """
    SELECT
        g.id AS grant_id,
        g.title,
        COUNT(d.id) AS total_documents,
        SUM(CASE WHEN d.doc_type = 'competition_section' THEN 1 ELSE 0 END) AS sections,
        SUM(CASE WHEN d.doc_type = 'briefing_pdf' THEN 1 ELSE 0 END) AS pdfs,
        SUM(CASE WHEN d.doc_type = 'guidance' THEN 1 ELSE 0 END) AS guidance,
        GROUP_CONCAT(DISTINCT d.doc_type) AS document_types,
        MAX(d.created_at) AS last_document_added
    FROM grants g
    LEFT JOIN documents d ON g.id = d.grant_id
    GROUP BY g.id, g.title
    ORDER BY g.id
    """

    df = pd.read_sql_query(query, conn)

    # Format date
    df['last_document_added'] = pd.to_datetime(df['last_document_added'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')

    return df


def export_corrections_sheet(conn):
    """
    Export audit log of all data corrections.

    Returns:
        pd.DataFrame: All manual and automatic corrections
    """
    query = """
    SELECT
        id AS grant_id,
        correction_type,
        CASE correction_type
            WHEN 'auto_title_cleanup' THEN old_total_fund
            ELSE NULL
        END AS old_title,
        CASE correction_type
            WHEN 'auto_title_cleanup' THEN new_total_fund
            ELSE NULL
        END AS new_title,
        CASE correction_type
            WHEN 'auto_title_cleanup' THEN NULL
            ELSE old_total_fund
        END AS old_funding_display,
        CASE correction_type
            WHEN 'auto_title_cleanup' THEN NULL
            ELSE new_total_fund
        END AS new_funding_display,
        old_total_fund_gbp AS old_funding_gbp,
        new_total_fund_gbp AS new_funding_gbp,
        reason,
        applied_at
    FROM manual_corrections
    ORDER BY applied_at DESC
    """

    df = pd.read_sql_query(query, conn)

    # Format date
    df['applied_at'] = pd.to_datetime(df['applied_at'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')

    # Add correction category
    df['category'] = df['correction_type'].map({
        'manual': 'Manual (Verified)',
        'automatic': 'Automatic (Heuristic)',
        'auto_decimal_from_docs': 'Decimal Refinement',
        'manual_prize_patch': 'Prize Competition',
        'auto_title_cleanup': 'Title Cleanup',
        'manual_decimal_override': 'Manual Override'
    })

    return df


def export_statistics_sheet(conn):
    """
    Export summary statistics and data quality metrics.

    Returns:
        pd.DataFrame: Database statistics in metric/value format
    """
    stats = []

    # Grant counts
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM grants")
    stats.append(["Total Grants", cursor.fetchone()[0]])

    cursor.execute("SELECT COUNT(*) FROM grants WHERE is_active = 1")
    stats.append(["Active Grants", cursor.fetchone()[0]])

    cursor.execute("SELECT COUNT(*) FROM grants WHERE is_active = 0 AND datetime(closes_at) < datetime('now')")
    stats.append(["Closed Grants", cursor.fetchone()[0]])

    cursor.execute("SELECT COUNT(*) FROM grants WHERE datetime(opens_at) > datetime('now')")
    stats.append(["Upcoming Grants", cursor.fetchone()[0]])

    cursor.execute("SELECT COUNT(*) FROM grants WHERE total_fund_gbp IS NOT NULL")
    stats.append(["Grants with Funding", cursor.fetchone()[0]])

    cursor.execute("SELECT COUNT(*) FROM grants WHERE total_fund_gbp IS NULL")
    stats.append(["Grants without Funding", cursor.fetchone()[0]])

    # Funding statistics
    cursor.execute("""
        SELECT
            ROUND(SUM(total_fund_gbp) / 1000000.0, 2),
            ROUND(AVG(total_fund_gbp) / 1000000.0, 2),
            ROUND(MIN(total_fund_gbp) / 1000000.0, 2),
            ROUND(MAX(total_fund_gbp) / 1000000.0, 2)
        FROM grants WHERE total_fund_gbp IS NOT NULL
    """)
    row = cursor.fetchone()
    if row and row[0]:
        total, avg, min_val, max_val = row
        stats.append(["Total Funding Available (£M)", total])
        stats.append(["Average Grant Size (£M)", avg])
        stats.append(["Smallest Grant (£M)", min_val])
        stats.append(["Largest Grant (£M)", max_val])

    # Document counts
    cursor.execute("SELECT COUNT(*) FROM documents")
    stats.append(["Total Documents", cursor.fetchone()[0]])

    cursor.execute("SELECT COUNT(*) FROM documents WHERE doc_type = 'competition_section'")
    stats.append(["Section Documents", cursor.fetchone()[0]])

    cursor.execute("SELECT COUNT(*) FROM documents WHERE doc_type = 'briefing_pdf'")
    stats.append(["PDF Documents", cursor.fetchone()[0]])

    cursor.execute("SELECT COUNT(DISTINCT grant_id) FROM documents")
    stats.append(["Grants with Documents", cursor.fetchone()[0]])

    # Embedding counts
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    stats.append(["Total Embeddings", cursor.fetchone()[0]])

    # Correction counts
    cursor.execute("SELECT COUNT(*) FROM manual_corrections")
    stats.append(["Total Corrections Applied", cursor.fetchone()[0]])

    cursor.execute("""
        SELECT correction_type, COUNT(*)
        FROM manual_corrections
        GROUP BY correction_type
    """)
    for corr_type, count in cursor.fetchall():
        stats.append([f"  - {corr_type}", count])

    # Data quality checks
    cursor.execute("""
        SELECT COUNT(*) FROM grants
        WHERE title LIKE 'Funding competition%'
    """)
    stats.append(["Title Prefix Issues", cursor.fetchone()[0]])

    cursor.execute("""
        SELECT COUNT(*) FROM grants
        WHERE total_fund_gbp IS NOT NULL
        AND total_fund_gbp < 100000
    """)
    stats.append(["Suspiciously Small Funding", cursor.fetchone()[0]])

    # Database metadata
    cursor.execute("SELECT datetime('now')")
    stats.append(["Export Date", cursor.fetchone()[0]])

    df = pd.DataFrame(stats, columns=["Metric", "Value"])
    return df


def export_to_excel(db_path, output_path, include_embeddings=False):
    """
    Export grants database to Excel workbook.

    Args:
        db_path: Path to SQLite database
        output_path: Path for output Excel file
        include_embeddings: If True, include embeddings summary sheet
    """
    print("=" * 80)
    print("📊 Exporting Database to Excel")
    print("=" * 80)
    print(f"Database: {db_path}")
    print(f"Output:   {output_path}")
    print("=" * 80)
    print()

    conn = sqlite3.connect(db_path)

    # Create Excel writer
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        print("📝 Exporting sheets...")

        # Sheet 1: Grants (main data)
        print("  - Grants (main data)")
        df_grants = export_grants_sheet(conn)
        df_grants.to_excel(writer, sheet_name="Grants", index=False)

        # Sheet 2: Document coverage summary
        print("  - Documents (coverage summary)")
        df_docs = export_documents_sheet(conn)
        df_docs.to_excel(writer, sheet_name="Document Coverage", index=False)

        # Sheet 3: Corrections audit log
        print("  - Manual Corrections (audit log)")
        df_corrections = export_corrections_sheet(conn)
        df_corrections.to_excel(writer, sheet_name="Corrections", index=False)

        # Sheet 4: Statistics
        print("  - Statistics (summary metrics)")
        df_stats = export_statistics_sheet(conn)
        df_stats.to_excel(writer, sheet_name="Statistics", index=False)

        # Sheet 5: Embeddings (optional)
        if include_embeddings:
            print("  - Embeddings (vector summary)")
            query = """
            SELECT
                e.document_id,
                d.grant_id,
                d.doc_type,
                e.model,
                e.created_at
            FROM embeddings e
            JOIN documents d ON e.document_id = d.id
            ORDER BY d.grant_id, e.document_id
            """
            df_embeddings = pd.read_sql_query(query, conn)
            df_embeddings['created_at'] = pd.to_datetime(df_embeddings['created_at'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            df_embeddings.to_excel(writer, sheet_name="Embeddings", index=False)

    conn.close()

    # Get file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB

    print()
    print("=" * 80)
    print("✅ Export Complete")
    print("=" * 80)
    print(f"File:     {output_path}")
    print(f"Size:     {file_size:.2f} MB")
    print(f"Sheets:   {4 + (1 if include_embeddings else 0)}")
    print("=" * 80)
    print()
    print("📋 SME Review Checklist:")
    print("  1. Open 'Grants' sheet - verify titles, funding, dates")
    print("  2. Check 'Document Coverage' - ensure all grants have docs")
    print("  3. Review 'Corrections' - validate all data changes")
    print("  4. Check 'Statistics' - confirm totals match expectations")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Export grants database to Excel workbook for SME review"
    )
    parser.add_argument(
        "--db",
        default="grants.db",
        help="Path to SQLite database (default: grants.db)"
    )
    parser.add_argument(
        "--out",
        default="grants_review.xlsx",
        help="Output Excel file path (default: grants_review.xlsx)"
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Include embeddings summary sheet (makes file larger)"
    )

    args = parser.parse_args()

    # Check if database exists
    if not Path(args.db).exists():
        print(f"❌ Database not found: {args.db}")
        return 1

    # Check dependencies
    try:
        import openpyxl
    except ImportError:
        print("❌ Missing dependency: openpyxl")
        print("   Install with: pip3 install openpyxl")
        return 1

    # Export
    export_to_excel(
        args.db,
        args.out,
        include_embeddings=args.include_embeddings
    )

    return 0


if __name__ == "__main__":
    exit(main())
