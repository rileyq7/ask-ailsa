# Ask Ailsa - UK Research Funding Discovery

AI-powered grant discovery platform for UK research funding from NIHR and Innovate UK. Features semantic search, conversational AI interface, and intelligent grant recommendations.

## Features

### Core Functionality
- **Semantic Search**: Vector-based search across NIHR and Innovate UK grants
- **Conversational AI**: GPT-4o-mini powered chat interface for natural language queries
- **Streaming Responses**: Real-time token-by-token response generation
- **Grant Recommendations**: Relevance-ranked results with scores and metadata
- **Smart Filtering**: Filter by deadline, funding amount, eligibility, and more

### SME Knowledge System
- **Expert Examples Database**: SQLite database storing SME curator expertise
- **Auto-Monitor Slack Bot**: Captures expert knowledge from Slack automatically
- **Quality Scoring**: Auto-assesses examples on 1-5 star scale
- **Multi-Format Capture**: Text, web links, PDFs, and Word documents
- **AI Integration**: Top examples injected into LLM prompts for better responses

### Data Extraction
- **Competition Metadata Extraction**: Titles, dates, funding rules, project sizes
- **Section Parsing**: Logical sections with fragment URLs (eligibility, scope, dates, etc.)
- **Resource Classification**: Automatic classification of resources as GLOBAL vs COMPETITION-SPECIFIC
- **Document Ingestion**: PDF text extraction and webpage parsing
- **De-duplication**: Content hashing to avoid duplicate processing

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Start the Backend API

```bash
./start_api.sh
```

The API will be available at `http://localhost:8000`

### 2. Start the Streamlit UI

```bash
./start_ui.sh
```

The UI will open automatically in your browser at `http://localhost:8501`

### 3. Start the Slack Bot (Optional)

Monitor your SME Slack channel for expert knowledge:

```bash
cd slack
./start_slack_bot.sh
```

The bot will auto-capture messages in `#sme-knowledge` without requiring @mentions. See [Slack Bot Setup](#slack-bot-setup) for configuration.

### Using Ask Ailsa

1. **Try Sample Questions**: Click any of the pre-made example questions
2. **Ask Custom Questions**: Type your query in the chat input and press Enter
3. **Review Results**: See AI-generated summaries with matched grant cards
4. **Explore Grants**: Click "View details →" to see full grant information

### Example Queries

- "Show me NIHR grants for clinical trials closing in the next 3 months"
- "Find Innovate UK competitions for AI and machine learning"
- "What funding is available for early-stage health technology research?"
- "Compare grant options for academic vs. commercial applicants"
- "Show me grants over £1M for medical device development"

### Run the Scraper Demo

```bash
python3 -m src.scripts.scrape_innovateuk_demo
```

### Use in Your Code

```python
from src.ingest.innovateuk_competition import InnovateUKCompetitionScraper
from src.ingest.resource_ingestor import ResourceIngestor

# Initialize scrapers
scraper = InnovateUKCompetitionScraper()
ingestor = ResourceIngestor()

# Scrape a competition page
url = "https://apply-for-innovation-funding.service.gov.uk/competition/2341/overview/..."
result = scraper.scrape_competition(url)

# Access structured data
print(f"Competition: {result.competition.title}")
print(f"Opens: {result.competition.opens_at}")
print(f"Closes: {result.competition.closes_at}")
print(f"Funding: {result.competition.total_fund}")

# Iterate through sections
for section in result.sections:
    print(f"Section: {section.name}")
    print(f"URL: {section.url}")
    print(f"Text: {section.text[:100]}...")

# Classify resources
for resource in result.resources:
    scope = "COMPETITION" if resource.competition_id else "GLOBAL"
    print(f"{scope} - {resource.type.value} - {resource.title}")

# Fetch and parse documents
docs = ingestor.fetch_documents_for_resources(result.resources)
for doc in docs:
    print(f"Document: {doc.doc_type} - {len(doc.text)} chars")
```

## Project Structure

```
.
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── .env                     # API configuration
├── grants.db               # SQLite database (grants + expert examples)
│
├── src/                    # Core application
│   ├── api/
│   │   └── server.py       # FastAPI backend with streaming
│   ├── core/
│   │   ├── models.py       # Data models
│   │   └── utils.py        # Utility functions
│   ├── ingest/
│   │   ├── innovateuk_competition.py   # Competition scraper
│   │   └── resource_ingestor.py        # Resource fetcher
│   └── scripts/
│       ├── run_api.py                  # API startup script
│       └── scrape_innovateuk_demo.py   # Scraper demo
│
├── ui/                     # Streamlit frontend
│   ├── app.py             # Main UI application
│   └── requirements.txt   # UI-specific dependencies
│
├── slack/                 # SME Knowledge Slack Bot
│   ├── sme_slack_bot.py          # Auto-monitoring Slack bot
│   ├── start_slack_bot.sh        # Bot launcher
│   ├── .env.slack               # Slack configuration
│   └── requirements-slack.txt   # Slack dependencies
│
├── scripts/               # Expert examples management
│   ├── create_expert_examples_table.py
│   ├── add_expert_example.py
│   ├── import_one_pager.py
│   ├── convert_docx_to_txt.py
│   └── view_expert_examples.py
│
├── tests/                 # Test suite
│   ├── test_basic_functionality.py
│   ├── test_nihr_scraper.py
│   ├── test_slack_connection.py
│   └── debug_slack_bot.py
│
├── sme_curations/         # Expert one-pager examples
│
└── Launcher scripts
    ├── start_api.sh       # Start backend API
    ├── start_ui.sh        # Start Streamlit UI
    └── start.sh           # Start all services
```

## Data Models

### Competition
- Metadata about a competition (title, dates, funding rules, etc.)
- Includes raw HTML for debugging/reparsing

### CompetitionSection
- Logical section of a competition page
- Each section has a fragment URL (e.g., `#eligibility`)

### SupportingResource
- PDF, video, or webpage linked from the competition
- Classified as GLOBAL (generic guidance) or COMPETITION (specific to this competition)

### Document
- Extracted text content from a resource
- Ready for vector search indexing

## Resource Classification

Resources are automatically classified based on URL patterns:

**COMPETITION-SPECIFIC** (linked to specific competition):
- URL contains competition ID
- URL is on apply-for-innovation-funding domain with `/competition/` path

**GLOBAL** (generic guidance):
- General UKRI/Innovate UK guidance pages
- Resources not specific to a single competition

## Slack Bot Setup

The Slack bot auto-monitors your SME channel and captures expert knowledge to improve AI responses.

### Initial Setup

1. **Install dependencies**:
   ```bash
   cd slack
   pip3 install -r requirements-slack.txt
   ```

2. **Configure Slack tokens** in `slack/.env.slack`:
   ```bash
   SLACK_BOT_TOKEN=xoxb-your-bot-token
   SLACK_APP_TOKEN=xapp-your-app-token
   SME_CHANNEL=sme-knowledge
   ```

3. **Create Slack App** at https://api.slack.com/apps
   - Enable Socket Mode
   - Add OAuth scopes: `channels:history`, `channels:read`, `chat:write`, `reactions:write`
   - Enable Event Subscriptions: Add `message.channels` bot event
   - Install to workspace

4. **Start the bot**:
   ```bash
   cd slack
   ./start_slack_bot.sh
   ```

### Bot Features

- **Auto-Monitoring**: Processes ALL messages (no @mention required)
- **Multi-Format**: Captures text, web links, PDFs, and Word documents
- **Quality Scoring**: Auto-assesses 1-5 stars based on content
- **Smart Detection**: Identifies grant names, categories, and strategies
- **Deduplication**: Prevents duplicate processing
- **Status Command**: Use `/sme-status` in Slack to see stats

### Managing Expert Examples

View captured examples:
```bash
python3 scripts/view_expert_examples.py
```

Add manual example:
```bash
python3 scripts/add_expert_example.py
```

Import from one-pager documents:
```bash
python3 scripts/import_one_pager.py sme_curations/example.txt
```

## Testing

### Test Core Functionality

```bash
python3 tests/test_basic_functionality.py
```

This tests:
- Utility functions (ID generation, hashing, date parsing)
- Data model creation
- Scraper components (ID extraction, scope classification, type inference)
- Resource ingestor components

### Test Slack Bot

```bash
python3 tests/debug_slack_bot.py
```

Checks authentication, channel access, and permissions.

### Unit Tests

```bash
pytest tests/ -v
```

## Implementation Notes

### Success Criteria Met

✅ Fetches Innovate UK competition pages
✅ Extracts structured metadata (title, dates, funding rules, project sizes)
✅ Slices content into logical sections with fragment URLs
✅ Classifies supporting resources as GLOBAL vs COMPETITION-SPECIFIC
✅ Fetches and parses PDFs/guidance documents
✅ Returns pure Python objects (NO database writes, NO LLM calls)

### Known Limitations

1. **Page Structure Variations**: The scraper is designed for a specific page structure. Some competitions may have different layouts that require adjustments to section detection logic.

2. **Section Detection**: The current implementation looks for specific section anchors (summary, eligibility, scope, etc.). If the page doesn't use these exact IDs/headers, sections won't be detected. This can be enhanced by:
   - Making section patterns more flexible
   - Using fuzzy matching for section names
   - Falling back to all h2/h3 headers if known sections aren't found

3. **Resource Links**: Resources are extracted from a "Supporting information" section. If this section isn't found, no resources will be extracted.

### Potential Enhancements

1. **Flexible Section Detection**: Detect all sections regardless of naming
2. **Multi-page Support**: Handle competitions split across multiple pages
3. **Caching**: Add HTTP response caching to avoid re-fetching
4. **Rate Limiting**: Add configurable rate limiting for politeness
5. **Retry Logic**: Exponential backoff for failed requests
6. **Progress Callbacks**: Report scraping progress for long operations

## Requirements

- Python 3.9+
- requests 2.31.0
- beautifulsoup4 4.12.3
- python-dateutil 2.8.2
- pdfplumber 0.11.0
- lxml 5.1.0

## License

[Specify your license here]

## Contact

[Your contact information]
