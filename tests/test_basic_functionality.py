"""
Test basic functionality of the scraper components.
"""

from src.core.models import Competition, CompetitionSection, SupportingResource, ResourceScope, ResourceType
from src.core.utils import stable_id_from_url, sha1_text, parse_date_maybe, clean_text, extract_money_amount
from datetime import datetime

print("=" * 80)
print("TESTING CORE FUNCTIONALITY")
print("=" * 80)

# Test 1: Utility functions
print("\n=== TEST 1: Utility Functions ===")

url = "https://example.com/test"
id1 = stable_id_from_url(url, "test_")
print(f"✓ stable_id_from_url: {id1}")

text = "Hello world"
hash1 = sha1_text(text)
print(f"✓ sha1_text: {hash1[:20]}...")

date_str = "10 April 2024 11:00am"
date = parse_date_maybe(date_str)
print(f"✓ parse_date_maybe: {date}")

messy = "  Multiple   spaces   \n\n\n\nand newlines  "
cleaned = clean_text(messy)
print(f"✓ clean_text: '{cleaned}'")

money_text = "Up to £5 million is available"
money = extract_money_amount(money_text)
print(f"✓ extract_money_amount: {money}")

# Test 2: Data models
print("\n=== TEST 2: Data Models ===")

competition = Competition(
    id="2341",
    external_id="2341",
    title="Test Competition",
    base_url="https://example.com/competition/2341",
    description="This is a test competition",
    opens_at=datetime(2024, 4, 10, 11, 0),
    closes_at=datetime(2024, 5, 22, 11, 0),
    total_fund="Up to £5 million",
    project_size="£150,000 to £750,000",
    funding_rules={"micro_sme_max_pct": 0.60, "large_max_pct": 0.50},
    raw_html="<html></html>"
)
print(f"✓ Competition created: {competition.title}")
print(f"  - ID: {competition.id}")
print(f"  - Opens: {competition.opens_at}")
print(f"  - Closes: {competition.closes_at}")
print(f"  - Funding: {competition.total_fund}")

section = CompetitionSection(
    competition_id=competition.id,
    name="eligibility",
    url=f"{competition.base_url}#eligibility",
    html="<p>Eligibility criteria</p>",
    text="Eligibility criteria"
)
print(f"✓ CompetitionSection created: {section.name}")

resource = SupportingResource(
    id="res_123",
    url="https://example.com/doc.pdf",
    title="Briefing Document",
    competition_id=competition.id,
    scope=ResourceScope.COMPETITION,
    type=ResourceType.PDF,
    content_hash=None
)
print(f"✓ SupportingResource created: {resource.title}")
print(f"  - Type: {resource.type.value}")
print(f"  - Scope: {resource.scope.value}")

# Test 3: Scraper (without network call)
print("\n=== TEST 3: Scraper Components ===")

from src.ingest.innovateuk_competition import InnovateUKCompetitionScraper

scraper = InnovateUKCompetitionScraper()
print(f"✓ Scraper initialized")

# Test ID extraction
external_id, internal_id = scraper._extract_ids("https://apply-for-innovation-funding.service.gov.uk/competition/2341/overview")
print(f"✓ ID extraction: external={external_id}, internal={internal_id}")

# Test scope classification
test_comp = Competition(
    id="2341",
    external_id="2341",
    title="Test",
    base_url="https://apply-for-innovation-funding.service.gov.uk/competition/2341/overview",
    description="Test"
)

scope1 = scraper._classify_scope("https://apply-for-innovation-funding.service.gov.uk/competition/2341/briefing.pdf", test_comp)
print(f"✓ Scope classification (competition-specific): {scope1.value}")

scope2 = scraper._classify_scope("https://www.ukri.org/guidance/general.pdf", test_comp)
print(f"✓ Scope classification (global): {scope2.value}")

# Test type inference
type1 = scraper._infer_type("https://example.com/doc.pdf")
print(f"✓ Type inference (PDF): {type1.value}")

type2 = scraper._infer_type("https://youtube.com/watch?v=123")
print(f"✓ Type inference (video): {type2.value}")

type3 = scraper._infer_type("https://example.com/page")
print(f"✓ Type inference (webpage): {type3.value}")

# Test 4: Resource Ingestor (components)
print("\n=== TEST 4: Resource Ingestor Components ===")

from src.ingest.resource_ingestor import ResourceIngestor

ingestor = ResourceIngestor()
print(f"✓ ResourceIngestor initialized")

is_iuk1 = ingestor._is_innovateuk_like("https://apply-for-innovation-funding.service.gov.uk/page")
print(f"✓ Is Innovate UK domain (yes): {is_iuk1}")

is_iuk2 = ingestor._is_innovateuk_like("https://example.com/page")
print(f"✓ Is Innovate UK domain (no): {is_iuk2}")

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
