#!/usr/bin/env python3
"""
Main runner script for DMBI Dashboard
Run this file to start the Streamlit dashboard
"""

import subprocess
import sys
import os

def main():
    """Run the main DMBI dashboard"""
    dashboard_path = os.path.join("src", "dashboard", "dmbi_dashboard.py")
    
    if not os.path.exists(dashboard_path):
        print("❌ Dashboard file not found!")
        print("Make sure you're running this from the project root directory.")
        return
    
    print("🚀 Starting DMBI Dashboard...")
    print("📊 Access the dashboard at: http://localhost:8502")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            "python3", "-m", "streamlit", "run", 
            dashboard_path, "--server.port", "8502"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped!")

if __name__ == "__main__":
    main()