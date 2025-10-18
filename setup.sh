#!/bin/bash
# Setup script for DMBI Mini Project

echo "🚀 Setting up DMBI Mini Project..."

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Activate environment: source .venv/bin/activate"
echo "   2. Run analysis: python3 run_analysis.py"
echo "   3. Start dashboard: python3 run_dashboard.py"