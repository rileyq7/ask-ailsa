#!/usr/bin/env python3
"""
Test Slack connection and permissions.
Run this to verify your Slack setup is working.
"""

import os
import sys

# Check environment variables
print("=" * 80)
print("🔍 SLACK CONNECTION TEST")
print("=" * 80)
print()

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
SME_CHANNEL = os.environ.get("SME_CHANNEL", "sme-knowledge")

print("1. Environment Variables:")
if SLACK_BOT_TOKEN:
    print(f"   ✅ SLACK_BOT_TOKEN: {SLACK_BOT_TOKEN[:15]}...")
else:
    print("   ❌ SLACK_BOT_TOKEN: Not set")

if SLACK_APP_TOKEN:
    print(f"   ✅ SLACK_APP_TOKEN: {SLACK_APP_TOKEN[:15]}...")
else:
    print("   ❌ SLACK_APP_TOKEN: Not set")

print(f"   📝 SME_CHANNEL: {SME_CHANNEL}")
print()

if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    print("❌ Missing tokens! Run: source .env.slack")
    sys.exit(1)

# Test Slack connection
print("2. Testing Slack API connection...")
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    client = WebClient(token=SLACK_BOT_TOKEN)

    # Test auth
    auth_response = client.auth_test()
    print(f"   ✅ Connected as: {auth_response['user']}")
    print(f"   ✅ Team: {auth_response['team']}")
    print()

    # List channels bot is in
    print("3. Checking channels...")
    channels_response = client.conversations_list(types="public_channel,private_channel")

    found_sme_channel = False
    bot_channels = []

    for channel in channels_response['channels']:
        # Check if bot is a member
        if channel.get('is_member'):
            bot_channels.append(channel['name'])
            if channel['name'] == SME_CHANNEL:
                found_sme_channel = True
                print(f"   ✅ Bot is in #{channel['name']} (ID: {channel['id']})")

    if not found_sme_channel:
        print(f"   ❌ Bot is NOT in #{SME_CHANNEL}")
        print()
        print("   Bot is currently in these channels:")
        for ch_name in bot_channels:
            print(f"      - #{ch_name}")
        print()
        print(f"   👉 Action needed: Invite bot to #{SME_CHANNEL}")
        print(f"      In Slack, type: /invite @SME Knowledge Bot")

    print()

    # Check permissions
    print("4. Checking bot permissions...")
    scopes = auth_response.get('scopes', [])

    required_scopes = ['channels:history', 'channels:read', 'chat:write', 'reactions:write']

    for scope in required_scopes:
        if scope in scopes:
            print(f"   ✅ {scope}")
        else:
            print(f"   ❌ {scope} - MISSING!")

    print()

    # Test Socket Mode
    print("5. Testing Socket Mode connection...")
    try:
        from slack_bolt import App
        from slack_bolt.adapter.socket_mode import SocketModeHandler

        app = App(token=SLACK_BOT_TOKEN)

        # Try to create handler (don't start it)
        handler = SocketModeHandler(app, SLACK_APP_TOKEN)
        print("   ✅ Socket Mode configured correctly")

    except Exception as e:
        print(f"   ❌ Socket Mode error: {e}")

    print()
    print("=" * 80)
    print("✅ CONNECTION TEST COMPLETE")
    print("=" * 80)
    print()

    if found_sme_channel and all(scope in scopes for scope in required_scopes):
        print("🎉 Everything looks good! Bot should work.")
        print()
        print("If messages still aren't being captured:")
        print("  1. Make sure messages are >50 characters")
        print("  2. Check bot console for errors")
        print("  3. Try restarting the bot")
    else:
        print("⚠️  Issues found - see above for details")

except ImportError:
    print("   ❌ slack_sdk not installed")
    print("   Run: pip3 install slack-sdk")
except SlackApiError as e:
    print(f"   ❌ Slack API Error: {e.response['error']}")
    if e.response['error'] == 'invalid_auth':
        print("   👉 Check your SLACK_BOT_TOKEN")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
