#!/bin/bash
set -e

echo "=" * 80
echo "🤖 Setting up SME Slack Bot"
echo "=" * 80
echo ""

# Check if slack-bolt is installed
if ! python3 -c "import slack_bolt" 2>/dev/null; then
    echo "📦 Installing Slack dependencies..."
    pip3 install slack-bolt slack-sdk --quiet
    echo "✅ Dependencies installed"
else
    echo "✅ Slack dependencies already installed"
fi

# Create .env.slack if it doesn't exist
if [ ! -f .env.slack ]; then
    echo ""
    echo "📝 Creating .env.slack configuration file..."
    cp .env.slack.example .env.slack
    echo "✅ Created .env.slack"
    echo ""
    echo "⚠️  ACTION REQUIRED:"
    echo "   Edit .env.slack and add your Slack tokens"
else
    echo "✅ .env.slack already exists"
fi

# Make bot script executable
chmod +x sme_slack_bot.py

echo ""
echo "=" * 80
echo "✅ Setup Complete!"
echo "=" * 80
echo ""
echo "Next steps:"
echo ""
echo "1. Create Slack App (if you haven't already)"
echo "   → Go to: https://api.slack.com/apps"
echo "   → Click 'Create New App' → 'From Scratch'"
echo "   → Name: 'SME Knowledge Bot'"
echo "   → Pick your workspace"
echo ""
echo "2. Configure Bot Permissions"
echo "   → Go to 'OAuth & Permissions'"
echo "   → Add Bot Token Scopes:"
echo "     - channels:history"
echo "     - channels:read"
echo "     - chat:write"
echo "     - reactions:write"
echo "   → Click 'Install to Workspace'"
echo "   → Copy 'Bot User OAuth Token' (starts with xoxb-)"
echo ""
echo "3. Enable Socket Mode"
echo "   → Go to 'Socket Mode'"
echo "   → Toggle 'Enable Socket Mode'"
echo "   → Go to 'Basic Information'"
echo "   → Create 'App-Level Token' with connections:write scope"
echo "   → Copy token (starts with xapp-)"
echo ""
echo "4. Add tokens to .env.slack"
echo "   → Edit .env.slack"
echo "   → Add both tokens"
echo ""
echo "5. Create Slack channel"
echo "   → Create #sme-knowledge channel in Slack"
echo "   → Invite the bot to the channel"
echo ""
echo "6. Start the bot"
echo "   → Run: source .env.slack && python3 sme_slack_bot.py"
echo ""
echo "7. Test it!"
echo "   → Post a message in #sme-knowledge"
echo "   → Bot should react with 🧠 emoji"
echo "   → Check with: python3 scripts/view_expert_examples.py"
echo ""
echo "=" * 80
