#!/bin/bash

# DMBI College Placement Analytics - Startup Script
echo "Starting DMBI College Placement Analytics Dashboard..."
echo "==============================================="

# Navigate to project directory
cd /home/parthnarkar/Desktop/DMBI-MiniProject

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Check if data file exists
if [ ! -f "data/placementdata.csv" ]; then
    echo "Error: placementdata.csv not found!"
    exit 1
fi

# Activate virtual environment and run dashboard
echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing/updating required packages..."
pip install -q streamlit pandas numpy matplotlib seaborn plotly scikit-learn scipy

echo "Starting Streamlit dashboard on port 8502..."
echo "Dashboard will be available at: http://localhost:8502"
echo "Press Ctrl+C to stop the dashboard"

# Run the dashboard
python -m streamlit run dmbi_dashboard.py --server.port 8502