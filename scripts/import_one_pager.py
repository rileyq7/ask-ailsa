#!/usr/bin/env python3
"""
Import full one-pager curations from SME.
Intelligently extracts multiple examples from a single document.
"""

import sqlite3
import uuid
import re
from datetime import datetime
from pathlib import Path
import sys

DB_PATH = Path(__file__).parent.parent / "grants.db"

sys.path.append(str(Path(__file__).parent.parent))
from scripts.add_expert_example import add_expert_example


def parse_one_pager(file_path: str):
    """
    Parse a one-pager document and extract individual grant/opportunity discussions.

    Returns list of extracted examples with metadata.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract client name/context from header if present
    client_context = extract_client_context(content)

    # Split into sections (each grant/opportunity)
    sections = split_into_sections(content)

    print(f"\n📄 Processing: {Path(file_path).name}")
    print(f"   Client: {client_context or 'Unknown'}")
    print(f"   Found {len(sections)} opportunities\n")

    examples = []
    for i, section in enumerate(sections, 1):
        example = parse_section(section, client_context, file_path)
        if example:
            examples.append(example)
            print(f"   ✓ Extracted example {i}/{len(sections)}: {example['grant'][:50]}...")

    return examples, client_context


def extract_client_context(content: str) -> str:
    """Extract client context from document header."""
    # Look for common patterns like "Hi [Name]," or client mentions
    first_lines = content[:500]

    # Pattern: "Hi Name," or "Dear Name,"
    greeting_match = re.search(r'Hi ([A-Z][a-z]+)', first_lines)
    if greeting_match:
        return f"client: {greeting_match.group(1)}"

    return None


def split_into_sections(content: str) -> list:
    """
    Split one-pager into individual grant/opportunity sections.

    Looks for common patterns:
    - "1. Grant Name" / "2. Grant Name"
    - "GRANT NAME" (all caps headers)
    - Double line breaks between major sections
    """
    sections = []

    # Try numbered sections first (most common pattern)
    numbered_sections = re.split(r'\n(?=\d+\.\s+[A-Z])', content)

    if len(numbered_sections) > 1:
        # Successfully split by numbered items
        # Skip the intro (before first number)
        for section in numbered_sections[1:]:
            if len(section.strip()) > 100:  # Only meaningful sections
                sections.append(section.strip())
    else:
        # Try splitting by major headers (all caps or with markers)
        header_splits = re.split(r'\n(?=[A-Z][A-Z\s]{10,})', content)
        if len(header_splits) > 1:
            for section in header_splits[1:]:
                if len(section.strip()) > 100:
                    sections.append(section.strip())
        else:
            # Fallback: treat as single section
            sections = [content]

    return sections


def parse_section(section: str, client_context: str, source_file: str) -> dict:
    """
    Parse a single grant/opportunity section.

    Extracts:
    - Grant name/title
    - Type of content (grant explanation, feasibility, strategy)
    - Key insights
    """
    # Extract grant name (usually first line or after number)
    lines = section.split('\n')
    first_line = lines[0].strip()

    # Clean up numbered prefix if present
    grant_name = re.sub(r'^\d+\.\s*', '', first_line)

    # Try to detect category based on content
    category = detect_category(section)

    # Generate a natural query that would lead to this response
    user_query = generate_query_from_section(section, grant_name)

    # Determine quality score based on content richness
    quality_score = assess_quality(section)

    return {
        'user_query': user_query,
        'expert_response': section,
        'category': category,
        'grant': grant_name,
        'client_context': client_context,
        'quality_score': quality_score,
        'source': Path(source_file).name
    }


def detect_category(section: str) -> str:
    """Detect the category of response based on content patterns."""
    section_lower = section.lower()

    # Check for key phrases
    if 'ailsa\'s take' in section_lower:
        # Full structured response - likely grant explanation
        if 'why it\'s a fit' in section_lower:
            return 'grant_explanation'

    if 'eligibility' in section_lower or 'eligible' in section_lower:
        return 'eligibility'

    if 'next steps' in section_lower or 'deadline' in section_lower:
        return 'application_process'

    if any(word in section_lower for word in ['position', 'strategy', 'approach', 'angle']):
        return 'positioning'

    if 'feasibility' in section_lower:
        return 'feasibility'

    if any(word in section_lower for word in ['loan', 'credit', 'repay']):
        return 'financing'

    return 'general'


def generate_query_from_section(section: str, grant_name: str) -> str:
    """Generate a natural user query that would lead to this response."""
    section_lower = section.lower()

    # Detect query type and generate appropriate question
    if 'feasibility' in section_lower:
        return f"What feasibility funding is available? Tell me about {grant_name}"

    if 'loan' in section_lower:
        return f"Should we consider innovation loans? What about {grant_name}?"

    if 'procurement' in section_lower or 'tender' in section_lower:
        return f"Are there any procurement opportunities for us? What about {grant_name}?"

    if 'partnership' in section_lower or 'network' in section_lower:
        return f"What networking or partnership opportunities should we explore?"

    # Default
    return f"Tell me about {grant_name}"


def assess_quality(section: str) -> int:
    """
    Assess quality of example (1-5).

    Criteria:
    - Has structure (Ailsa's Take, Why it's a fit, Next steps) = +2
    - Has specific numbers (amounts, dates, percentages) = +1
    - Has strategic advice = +1
    - Length and detail = +1
    """
    score = 2  # Base score

    section_lower = section.lower()

    # Check for structured format
    if all(phrase in section_lower for phrase in ['ailsa\'s take', 'why it\'s a fit', 'next steps']):
        score += 2
    elif any(phrase in section_lower for phrase in ['ailsa\'s take', 'why it\'s a fit']):
        score += 1

    # Check for specific numbers (£ amounts, percentages, dates)
    if re.search(r'£[\d,]+[KM]?', section):
        score += 0.5
    if re.search(r'\d+%', section):
        score += 0.5

    # Check for strategic keywords
    strategic_words = ['position', 'angle', 'transform', 'bold', 'strategy', 'leverage']
    if any(word in section_lower for word in strategic_words):
        score += 0.5

    # Length bonus
    if len(section) > 500:
        score += 0.5

    return min(5, round(score))


def batch_import_one_pagers(directory: str):
    """
    Import all one-pagers from a directory.

    Usage:
        python scripts/import_one_pager.py one_pagers/
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        print(f"❌ Directory not found: {directory}")
        return

    # Find all text files
    files = list(directory_path.glob('*.txt')) + list(directory_path.glob('*.md'))

    if not files:
        print(f"❌ No .txt or .md files found in {directory}")
        return

    print(f"\n🚀 Found {len(files)} one-pager(s) to process\n")
    print("="*80)

    total_examples = 0

    for file_path in files:
        try:
            examples, client_context = parse_one_pager(str(file_path))

            # Add each example to database
            for example in examples:
                add_expert_example(
                    user_query=example['user_query'],
                    expert_response=example['expert_response'],
                    category=example['category'],
                    client_context=example['client_context'],
                    grant_mentioned=example['grant'],
                    notes=f"From one-pager: {example['source']}",
                    quality_score=example['quality_score']
                )
                total_examples += 1

            print(f"\n   ✅ Imported {len(examples)} examples from {file_path.name}")

        except Exception as e:
            print(f"\n   ❌ Error processing {file_path.name}: {e}")
            continue

    print("\n" + "="*80)
    print(f"🎉 Successfully imported {total_examples} examples from {len(files)} one-pagers!")
    print("\nNext steps:")
    print("  1. Review examples: python scripts/view_expert_examples.py")
    print("  2. Restart backend to load new examples")
    print("  3. Test the new SME voice!")


def process_single_file(file_path: str):
    """Process a single one-pager file interactively."""
    examples, client_context = parse_one_pager(file_path)

    print("\n" + "="*80)
    print("REVIEW EXTRACTED EXAMPLES")
    print("="*80)
    print(f"\nExtracted {len(examples)} examples. Review and confirm each:\n")

    confirmed = 0

    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i}/{len(examples)} ---")
        print(f"Grant: {example['grant']}")
        print(f"Category: {example['category']}")
        print(f"Quality: {example['quality_score']}/5")
        print(f"Query: {example['user_query']}")
        print(f"\nResponse preview:\n{example['expert_response'][:300]}...")

        choice = input(f"\nImport this example? (y/n/e=edit/s=skip all): ").strip().lower()

        if choice == 's':
            print("⏭️  Skipping remaining examples")
            break
        elif choice == 'e':
            # Edit metadata
            example['category'] = input(f"Category [{example['category']}]: ").strip() or example['category']
            example['user_query'] = input(f"Query [{example['user_query']}]: ").strip() or example['user_query']
            example['quality_score'] = int(input(f"Quality [{example['quality_score']}]: ").strip() or example['quality_score'])
            choice = 'y'

        if choice == 'y':
            add_expert_example(
                user_query=example['user_query'],
                expert_response=example['expert_response'],
                category=example['category'],
                client_context=example['client_context'],
                grant_mentioned=example['grant'],
                notes=f"From one-pager: {Path(file_path).name}",
                quality_score=example['quality_score']
            )
            confirmed += 1
            print("✅ Imported")
        else:
            print("⏭️  Skipped")

    print(f"\n🎉 Imported {confirmed}/{len(examples)} examples")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Batch import:   python scripts/import_one_pager.py path/to/one_pagers/")
        print("  Single file:    python scripts/import_one_pager.py path/to/file.txt")
        sys.exit(1)

    path = sys.argv[1]

    if Path(path).is_dir():
        # Batch mode
        batch_import_one_pagers(path)
    else:
        # Single file mode
        process_single_file(path)
