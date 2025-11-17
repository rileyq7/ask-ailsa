#!/bin/bash
# SME Slack Bot Launcher
# Loads environment variables and starts the auto-monitoring Slack bot

# Check if .env.slack exists
if [ ! -f .env.slack ]; then
    echo "❌ .env.slack not found!"
    echo ""
    echo "Create it from the template:"
    echo "  cp .env.slack.example .env.slack"
    echo ""
    echo "Then add your Slack tokens to .env.slack"
    exit 1
fi

# Load environment variables
echo "📝 Loading configuration from .env.slack..."
set -a
source .env.slack
set +a

# Verify tokens are set
if [ -z "$SLACK_BOT_TOKEN" ] || [ -z "$SLACK_APP_TOKEN" ]; then
    echo "❌ Tokens not configured in .env.slack"
    echo ""
    echo "Edit .env.slack and add your tokens:"
    echo "  SLACK_BOT_TOKEN=xoxb-your-token"
    echo "  SLACK_APP_TOKEN=xapp-your-token"
    exit 1
fi

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import slack_bolt; import requests; import bs4" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies"
    echo ""
    echo "Install with:"
    echo "  pip3 install -r requirements-slack.txt"
    exit 1
fi

echo "✅ Dependencies OK"
echo ""

# Start the bot
echo "🚀 Starting SME Slack Bot..."
echo ""
python3 sme_slack_bot.py
