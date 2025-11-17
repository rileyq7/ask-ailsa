#!/bin/bash

echo "🚀 Setting up SME Knowledge System..."

# Create database table
python scripts/create_expert_examples_table.py

# Create directory for one-pagers
mkdir -p sme_curations

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy your SME one-pagers to: sme_curations/"
echo "  2. Run: python scripts/import_one_pager.py sme_curations/"
echo "  3. View imported examples: python scripts/view_expert_examples.py"
echo "  4. Restart backend: ./start_api.sh"
echo ""
