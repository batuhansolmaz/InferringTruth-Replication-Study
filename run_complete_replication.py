import subprocess
import sys
import os
from datetime import datetime

def run_human_experiment_analysis():
    """Run the human experiment data analysis"""
    print("Running human experiment analysis...")
    
    try:
        result = subprocess.run([sys.executable, "comprehensive_analysis.py"], 
                              capture_output=True, text=True, check=True)
        print("✓ Human experiment analysis completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in human experiment analysis: {e}")
        return False

def run_computational_simulations():
    """Run the computational agent-based simulations"""
    print("Running computational simulations...")
    
    try:
        result = subprocess.run([sys.executable, "computational_simulations.py"], 
                              capture_output=True, text=True, check=True)
        print("✓ Computational simulations completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in computational simulations: {e}")
        return False

def main():
    """Run the complete replication study"""
    print("=" * 50)
    print("OEY & VUL (2024) REPLICATION")
    print("=" * 50)
    
    # Run both parts
    human_success = run_human_experiment_analysis()
    computational_success = run_computational_simulations()
    
    if human_success and computational_success:
        print("\n🎉 Replication completed successfully")
    else:
        print("\n⚠ Replication completed with issues")
    
    print(f"Finished at: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main() 