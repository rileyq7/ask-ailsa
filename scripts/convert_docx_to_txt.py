#!/usr/bin/env python3
"""
Convert Word documents (.docx) to plain text for import.

Usage:
    python3 scripts/convert_docx_to_txt.py sme_curations/
"""

import sys
from pathlib import Path

try:
    from docx import Document
except ImportError:
    print("❌ python-docx not installed")
    print("\nInstall it with:")
    print("  pip install python-docx")
    print("\nOr use manual conversion:")
    print("  1. Open Word doc")
    print("  2. Save As → Plain Text (.txt)")
    print("  3. Save to sme_curations/")
    sys.exit(1)


def convert_docx_to_txt(docx_path: Path) -> Path:
    """
    Convert a .docx file to plain text.

    Args:
        docx_path: Path to .docx file

    Returns:
        Path to created .txt file
    """
    try:
        # Read Word document
        doc = Document(docx_path)

        # Extract all paragraphs
        text_content = []
        for para in doc.paragraphs:
            if para.text.strip():  # Skip empty paragraphs
                text_content.append(para.text)

        # Create .txt file path
        txt_path = docx_path.with_suffix('.txt')

        # Write to text file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(text_content))

        print(f"✅ Converted: {docx_path.name} → {txt_path.name}")
        return txt_path

    except Exception as e:
        print(f"❌ Failed to convert {docx_path.name}: {e}")
        return None


def convert_directory(directory: str):
    """Convert all .docx files in a directory to .txt."""
    directory_path = Path(directory)

    if not directory_path.exists():
        print(f"❌ Directory not found: {directory}")
        return

    # Find all .docx files
    docx_files = list(directory_path.glob('*.docx'))

    if not docx_files:
        print(f"⚠️  No .docx files found in {directory}")
        return

    print(f"\n🔄 Found {len(docx_files)} Word document(s) to convert\n")
    print("=" * 60)

    converted = 0

    for docx_file in docx_files:
        # Skip temporary Word files
        if docx_file.name.startswith('~$'):
            print(f"⏭️  Skipping temporary file: {docx_file.name}")
            continue

        txt_file = convert_docx_to_txt(docx_file)
        if txt_file:
            converted += 1

    print("=" * 60)
    print(f"\n🎉 Converted {converted}/{len(docx_files)} files successfully!")

    if converted > 0:
        print("\nNext steps:")
        print(f"  1. Review converted .txt files in {directory}")
        print(f"  2. Run: python3 scripts/import_one_pager.py {directory}")
        print("  3. View imported examples: python3 scripts/view_expert_examples.py")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 scripts/convert_docx_to_txt.py sme_curations/")
        print("\nConverts all .docx files in directory to .txt for import")
        sys.exit(1)

    directory = sys.argv[1]
    convert_directory(directory)
