#!/usr/bin/env python3
"""
Ask Ailsa - AI-powered UK Research Funding Discovery
FIXED VERSION - Proper streaming and markdown rendering
"""

import streamlit as st
import requests
import json
import time
from typing import List, Dict, Iterable
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:8000"
STREAM_ENDPOINT = f"{BACKEND_URL}/chat/stream"

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ask Ailsa",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
        /* Main header styling */
        .ask-ailsa-header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        .ask-ailsa-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #f1f5f9;
            margin: 0;
            margin-bottom: 0.5rem;
        }
        .ask-ailsa-subtitle {
            font-size: 1.1rem;
            color: #94a3b8;
            margin: 0;
        }

        /* Grant card styling */
        .grant-card {
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-left: 4px solid #6366f1;
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin: 1rem 0;
            background: rgba(30, 41, 59, 0.4);
            transition: all 0.2s;
        }
        .grant-card:hover {
            border-left-color: #818cf8;
            background: rgba(30, 41, 59, 0.6);
        }
        .grant-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #f1f5f9;
            margin-bottom: 0.75rem;
        }
        .grant-meta {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        .grant-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            color: #cbd5e1;
        }
        .grant-source {
            background: #6366f1;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: 600;
        }

        /* Chat message styling */
        .stChatMessage h3 {
            font-size: 1rem;
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
            padding-top: 0.5rem !important;
            color: #e2e8f0 !important;
        }

        .stChatMessage ul, .stChatMessage ol {
            font-size: 0.9rem;
            margin-left: 1.5rem !important;
            margin-bottom: 1rem !important;
        }

        .stChatMessage li {
            margin-bottom: 0.5rem !important;
            line-height: 1.6 !important;
            color: #cbd5e1 !important;
        }

        .stChatMessage strong {
            font-weight: 600;
            color: #e2e8f0 !important;
        }

        hr {
            margin: 1.5rem 0 !important;
            border-color: rgba(148, 163, 184, 0.2) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict] = [
        {
            "role": "assistant",
            "content": "Hi, I'm Ailsa 👋\n\nI can help you discover UK research funding from NIHR and Innovate UK. Ask me about specific grants, deadlines, or funding amounts, or try one of the examples below.",
            "grants": []
        }
    ]

if "pending_sample_question" not in st.session_state:
    st.session_state.pending_sample_question = None

# ─────────────────────────────────────────────────────────────
# BACKEND COMMUNICATION
# ─────────────────────────────────────────────────────────────

def ask_ailsa_stream(user_prompt: str) -> Iterable[Dict]:
    """
    Stream response from Ask Ailsa backend.
    Yields dictionaries with 'type' key: 'token', 'grants', 'error', or 'done'.
    """
    try:
        logger.info(f"Sending request to {STREAM_ENDPOINT}")

        # Build conversation history from session state (last 10 messages = 5 exchanges)
        history = []
        for msg in st.session_state.messages[-10:]:
            if msg["role"] in ["user", "assistant"]:
                history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        logger.info(f"Sending {len(history)} historical messages for context")

        response = requests.post(
            STREAM_ENDPOINT,
            json={"message": user_prompt, "history": history, "active_only": True, "sources": None},
            stream=True,
            timeout=120,
            headers={"Accept": "text/event-stream"},  # Explicitly request SSE
        )
        response.raise_for_status()

        logger.info("Starting to process SSE stream")

        # Use iter_content instead of iter_lines to avoid line buffering
        buffer = ""
        for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
            if chunk:
                buffer += chunk

                # Look for complete SSE messages (end with \n\n)
                while "\n\n" in buffer:
                    message, buffer = buffer.split("\n\n", 1)

                    # Parse SSE format
                    for line in message.split("\n"):
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                data = json.loads(data_str)
                                logger.debug(f"Received chunk: {data.get('type')}")
                                yield data
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON decode error: {e}, line: {data_str}")
                                continue

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        yield {"type": "error", "error": f"Failed to connect to Ailsa backend: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        yield {"type": "error", "error": f"Unexpected error: {str(e)}"}


def render_grant_card(grant: Dict):
    """Render a single grant card with proper styling."""
    title = grant.get("title", "Untitled Grant")
    source = grant.get("source", "unknown").upper().replace("_", " ")
    funding = grant.get("total_fund_gbp")
    deadline = grant.get("closes_at", "").split("T")[0] if grant.get("closes_at") else "No deadline"
    url = grant.get("url", "#")

    # Format funding - handle both numeric and string values safely
    funding_str = "Funding TBC"
    if funding and isinstance(funding, (int, float)):
        if funding >= 1_000_000:
            funding_str = f"£{funding/1_000_000:.1f}M"
        elif funding >= 1_000:
            funding_str = f"£{funding/1_000:.0f}K"
        else:
            funding_str = f"£{funding:,.0f}"

    st.markdown(
        f"""
        <div class="grant-card">
            <div class="grant-title">{title}</div>
            <div class="grant-meta">
                <span class="grant-source">{source}</span>
                <span class="grant-badge">💰 {funding_str}</span>
                <span class="grant-badge">📅 Closes: {deadline}</span>
            </div>
            <a href="{url}" target="_blank" style="color: #6366f1; text-decoration: none;">View details →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def handle_user_message(user_text: str):
    """Process user message and stream response with proper formatting."""
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "grants": []
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_text)

    # Stream assistant response
    with st.chat_message("assistant"):
        full_response = ""
        grants_data = []

        # Container for streaming text
        text_placeholder = st.empty()
        error_occurred = False

        try:
            # Stream the response
            for chunk in ask_ailsa_stream(user_text):
                chunk_type = chunk.get("type")
                
                if chunk_type == "error":
                    st.error(chunk.get("error", "Unknown error"))
                    full_response = f"❌ {chunk.get('error', 'Unknown error')}"
                    error_occurred = True
                    break

                elif chunk_type == "token":
                    # Append token to response
                    content = chunk.get("content", "")
                    full_response += content
                    # Update placeholder with streaming text + cursor
                    text_placeholder.markdown(full_response + " ●")
                    # Small delay to force Streamlit to render (feels natural)
                    time.sleep(0.005)

                elif chunk_type == "grants":
                    grants_data = chunk.get("grants", [])

                elif chunk_type == "done":
                    break

            # Clear the placeholder
            text_placeholder.empty()

            # Render final formatted version
            if full_response and not error_occurred:
                st.markdown(full_response)
            elif not full_response and not error_occurred:
                st.markdown("_No response received from the backend._")

            # Render grant cards
            if grants_data:
                st.markdown("---")
                st.markdown("### 📋 Matched Grants")
                for grant in grants_data:
                    render_grant_card(grant)

        except Exception as e:
            logger.error(f"Error in handle_user_message: {e}")
            st.error(f"An error occurred: {str(e)}")
            full_response = f"❌ Error: {str(e)}"

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "grants": grants_data
    })

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────

# Sidebar
with st.sidebar:
    st.markdown("### About Ask Ailsa")
    st.markdown(
        """
        **Ask Ailsa** helps you discover UK research funding through conversational AI.
        
        - 🔍 Semantic search across NIHR & Innovate UK
        - 💬 Natural language queries
        - 📊 Relevance-ranked results
        - 🎯 Smart filtering by deadline, amount, eligibility
        """
    )

    st.markdown("---")
    st.markdown("### Tips")
    st.markdown(
        """
        - Be specific about your research area
        - Ask about deadlines or funding amounts
        - Request comparisons between grants
        - Follow up to refine results
        """
    )

    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("🏥 NIHR Funding  \n💡 Innovate UK Competitions")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. How can I help you find funding?",
                "grants": []
            }
        ]
        st.rerun()

# Header
st.markdown(
    """
    <div class="ask-ailsa-header">
        <h1 class="ask-ailsa-title">🔬 Ask Ailsa</h1>
        <p class="ask-ailsa-subtitle">Your AI guide to UK research funding opportunities</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sample questions
st.markdown("#### Try asking:")
cols = st.columns(3)

sample_questions = [
    "Show me NIHR grants for clinical trials closing in the next 3 months",
    "Find Innovate UK competitions for AI and machine learning",
    "What funding is available for early-stage health technology research?",
    "Compare grant options for academic vs. commercial applicants",
    "Show me grants over £1M for medical device development",
    "What NIHR i4i programs are currently open?",
]

for i, question in enumerate(sample_questions):
    col = cols[i % 3]
    with col:
        if st.button(question, key=f"sample-{i}", use_container_width=True):
            st.session_state.pending_sample_question = question

st.markdown("---")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Render grants if this message has them
        if msg.get("grants"):
            st.divider()
            st.markdown("### 📋 Matched Grants")
            for grant in msg["grants"]:
                render_grant_card(grant)

# Handle pending sample question
if st.session_state.pending_sample_question:
    q = st.session_state.pending_sample_question
    st.session_state.pending_sample_question = None
    handle_user_message(q)
    st.rerun()

# Chat input
user_input = st.chat_input("Describe your project or ask about funding...")

if user_input:
    handle_user_message(user_input)
    st.rerun()
