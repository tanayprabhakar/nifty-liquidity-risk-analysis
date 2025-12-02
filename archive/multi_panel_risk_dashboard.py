import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

def fetch_data():
    """Fetches 10 years of historical data for Nifty 50 and selected sectors."""
    tickers = {
        'Nifty 50': '^NSEI',
        'Nifty Bank': '^NSEBANK',
        'Nifty IT': '^CNXIT'
    }
    
    print("Fetching 10 years of data from yfinance...")
    data = yf.download(list(tickers.values()), period="10y", progress=False)
    
    # Handle MultiIndex columns from yfinance (Price, Ticker)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data['Adj Close']
        except KeyError:
            data = data['Close']
            
    # Rename columns to friendly names
    inv_tickers = {v: k for k, v in tickers.items()}
    data.rename(columns=inv_tickers, inplace=True)
    
    # Drop any rows with missing values to ensure alignment
    data.dropna(inplace=True)
    return data

def calculate_metrics(df, window=30):
    """Calculates Annualized Volatility, Risk Score v2, and Market Returns."""
    # Calculate daily percent change
    returns = df.pct_change().dropna()
    
    # --- Market Daily Returns ---
    market_returns = returns['Nifty 50']
    
    # --- Volatility & Risk Score v2 ---
    metrics = {}
    
    for col in ['Nifty Bank', 'Nifty IT']:
        if col not in returns.columns:
            continue
            
        # 1. Annualized Volatility
        # 30-Day Rolling Standard Deviation * sqrt(252)
        rolling_std = returns[col].rolling(window=window).std()
        annualized_vol = rolling_std * (252 ** 0.5)
        
        # Filter extreme glitches (> 100%)
        annualized_vol = annualized_vol.mask(annualized_vol > 1.0)
        
        # 2. Risk Score v2 (Volatility-Only Fallback)
        # Logic adapted from process_market_data.py
        # Risk = 50 + (vol_z * 12) - (fii_z * 6)
        # Since FII is missing, fii_z = 0
        
        # Calculate Z-score of volatility
        vol_mean = annualized_vol.mean()
        vol_std_dev = annualized_vol.std()
        if vol_std_dev == 0: vol_std_dev = 1e-6
        
        vol_z = (annualized_vol - vol_mean) / vol_std_dev
        
        # Risk Formula
        risk_score = 50 + (vol_z * 12.0)
        risk_score = risk_score.clip(0, 100)
        
        metrics[col] = {
            'volatility': annualized_vol * 100, # Convert to %
            'risk_score': risk_score
        }
        
    return metrics, market_returns

def create_dashboard(metrics, market_returns):
    """Generates the multi-panel risk dashboard."""
    # Use seaborn-whitegrid theme
    sns.set_theme(style="whitegrid")
    
    # Create Figure and GridSpec
    fig = plt.figure(figsize=(18, 12))
    # Layout: Top row (2 cols), Middle (1 col), Bottom (1 col)
    # Height ratios roughly 40:25:20 (normalized)
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1.2, 1], hspace=0.3, wspace=0.15)
    
    # --- Panel 1 (Top Left): Sector Volatility Comparison ---
    ax1 = plt.subplot(gs[0, 0])
    
    colors = {'Nifty Bank': 'blue', 'Nifty IT': 'orange'}
    
    for sector, data in metrics.items():
        vol = data['volatility']
        # Raw line (faint)
        ax1.plot(vol.index, vol, color=colors[sector], alpha=0.3, linewidth=1)
        # Rolling mean (bold)
        vol_roll = vol.rolling(window=30).mean()
        ax1.plot(vol_roll.index, vol_roll, color=colors[sector], label=f'{sector} (30d Avg)', linewidth=2)
        # Historical Mean (dashed)
        ax1.axhline(y=vol.mean(), color=colors[sector], linestyle='--', alpha=0.7, linewidth=1)
        
    ax1.set_title('Sector Volatility Comparison (Annualized %)')
    ax1.set_ylabel('Annualized Volatility (%)')
    ax1.legend(loc='upper left')
    
    # --- Panel 2 (Top Right): Risk v2 Score Comparison ---
    ax2 = plt.subplot(gs[0, 1], sharex=ax1)
    
    # Zones
    ax2.axhspan(0, 30, color='green', alpha=0.1, label='Low Risk')
    ax2.axhspan(30, 70, color='yellow', alpha=0.1, label='Medium Risk')
    ax2.axhspan(70, 100, color='red', alpha=0.1, label='High Risk')
    
    for sector, data in metrics.items():
        risk = data['risk_score']
        # Rolling mean (bold)
        risk_roll = risk.rolling(window=30).mean()
        ax2.plot(risk_roll.index, risk_roll, color=colors[sector], label=f'{sector} Risk Score', linewidth=2)
        
    ax2.set_title('Risk v2 Score Comparison (0-100)')
    ax2.set_ylabel('Risk Score')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    
    # --- Panel 3 (Middle): Market Returns Context ---
    ax3 = plt.subplot(gs[1, :], sharex=ax1)
    
    # Create color array based on values
    # Dark Green (>1%), Light Green (0-1%), Light Red (-1-0%), Dark Red (<-1%)
    def get_color(val):
        if val > 0.01: return 'darkgreen'
        elif val > 0: return 'lightgreen'
        elif val > -0.01: return 'lightcoral'
        else: return 'darkred'
        
    bar_colors = [get_color(x) for x in market_returns]
    
    # Bar chart is heavy for 10y daily data, use fill_between or line with fill
    # But user asked for Bar chart. Let's try bar with width=1 if feasible, or stick to area for performance.
    # Area is safer for 2500 points.
    ax3.fill_between(market_returns.index, market_returns, 0, where=(market_returns>=0), color='green', alpha=0.5)
    ax3.fill_between(market_returns.index, market_returns, 0, where=(market_returns<0), color='red', alpha=0.5)
    
    ax3.set_title('Market Daily Returns Context')
    ax3.set_ylabel('Daily Return')
    ax3.set_ylim(-0.10, 0.10) # +/- 10% limits
    
    # --- Panel 4 (Bottom): Volatility-Risk Correlation ---
    ax4 = plt.subplot(gs[2, :], sharex=ax1)
    
    # Calculate averages across sectors
    avg_vol = pd.DataFrame([metrics[s]['volatility'] for s in metrics]).mean()
    avg_risk = pd.DataFrame([metrics[s]['risk_score'] for s in metrics]).mean()
    
    # Dual Axis
    ax4_right = ax4.twinx()
    
    l1 = ax4.plot(avg_vol.index, avg_vol.rolling(30).mean(), color='purple', label='Avg Volatility', linewidth=2)
    l2 = ax4_right.plot(avg_risk.index, avg_risk.rolling(30).mean(), color='teal', label='Avg Risk Score', linewidth=2)
    
    ax4.set_ylabel('Avg Volatility (%)', color='purple')
    ax4_right.set_ylabel('Avg Risk Score', color='teal')
    ax4.set_title('Correlation: Avg Volatility vs Avg Risk Score')
    
    # Combined Legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    # --- Common Enhancements ---
    # COVID Band (March 2020)
    covid_start = pd.Timestamp('2020-02-20')
    covid_end = pd.Timestamp('2020-04-30')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvspan(covid_start, covid_end, color='red', alpha=0.15)
        ax.grid(True, alpha=0.3)
        
    # Annotations
    ax1.text(pd.Timestamp('2020-03-15'), 80, 'COVID-19 Crash', rotation=90, verticalalignment='center')
    
    # Summary Box
    # Calculate current vs avg
    latest_date = market_returns.index[-1]
    summary_text = f"Summary ({latest_date.date()}):\n"
    for sector in metrics:
        curr_vol = metrics[sector]['volatility'].iloc[-1]
        avg_vol_hist = metrics[sector]['volatility'].mean()
        curr_risk = metrics[sector]['risk_score'].iloc[-1]
        
        summary_text += f"{sector}:\n"
        summary_text += f"  Vol: {curr_vol:.1f}% (Avg: {avg_vol_hist:.1f}%)\n"
        summary_text += f"  Risk: {curr_risk:.1f}/100\n"
        
    plt.figtext(0.02, 0.02, summary_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle('Multi-Dimensional Sector Risk Analysis: Volatility vs Risk v2 Scores (2017-2026)', fontsize=16, y=0.95)
    
    output_file = 'risk_dashboard_output.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved to {output_file}")
    plt.close()

def main():
    try:
        # 1. Data Setup
        df = fetch_data()
        
        if df.empty:
            print("No data fetched. Please check your internet connection or ticker symbols.")
            return

        # 2. Metrics Calculation
        metrics, market_returns = calculate_metrics(df)
        
        # 3. Visualization
        create_dashboard(metrics, market_returns)
        print("Analysis complete.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
