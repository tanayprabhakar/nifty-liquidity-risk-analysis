import subprocess
import sys
import os

def run(script):
    print(f"running {script}...")
    try:
        # run from src dir so relative paths work
        res = subprocess.run([sys.executable, script], cwd="src", check=True, capture_output=True, text=True)
        print(res.stdout)
        print(f"ok {script}")
    except subprocess.CalledProcessError as e:
        print(f"error {script}:")
        print(e.stderr)
        sys.exit(1)

def main():
    print("starting analysis...")
    
    # collect data (optional, uncomment to run)
    # run("collect_data.py")

    # process data
    run("process_market_data.py")
    
    # charts
    scripts = [
        "fii_leadlag_analysis.py",
        "market_risk_regimes.py",
        "rolling_beta_30d_analysis.py",
        "sector_momentum_analysis.py",
        "sector_correlation_matrix.py"
    ]
    
    for s in scripts:
        if os.path.exists(os.path.join("src", s)):
            run(s)
        else:
            print(f"missing src/{s}")
            
    print("\nall done.")

if __name__ == "__main__":
    main()
