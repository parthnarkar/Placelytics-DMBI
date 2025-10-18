#!/usr/bin/env python3
"""
Run all analysis scripts
"""

import subprocess
import sys
import os

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n🔄 Running {description}...")
    print("-" * 50)
    
    try:
        result = subprocess.run(["python3", script_path], 
                              capture_output=False, 
                              text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
    except Exception as e:
        print(f"❌ Error running {description}: {e}")

def main():
    """Run all analysis components"""
    print("🚀 DMBI Analysis Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("data/placementdata.csv"):
        print("❌ Please run this script from the project root directory!")
        return
    
    scripts = [
        ("src/analysis/placement_analysis.py", "Basic Placement Analysis"),
        ("src/analysis/advanced_dmbi_analysis.py", "Advanced DMBI Analysis"),
        ("src/validation/quick_validation.py", "Quick Model Validation"),
        ("src/analysis/create_visualizations.py", "Visualization Generation")
    ]
    
    for script_path, description in scripts:
        if os.path.exists(script_path):
            run_script(script_path, description)
        else:
            print(f"⚠️ Warning: {script_path} not found, skipping...")
    
    print("\n🎉 Analysis Suite Complete!")
    print("📊 Start the dashboard with: python run_dashboard.py")

if __name__ == "__main__":
    main()