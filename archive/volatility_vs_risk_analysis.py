import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def fetch_data():
    """Fetches NIFTY 50 data from 2015 to present."""
    print("Fetching NIFTY 50 data from 2015-01-01 to 2025-11-28...")
    data = yf.download('^NSEI', start='2015-01-01', end='2025-11-28', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
            close = data['Adj Close']
        except KeyError:
            close = data['Close']
    else:
        close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
        
    df = pd.DataFrame({'Close': close})
    df.dropna(inplace=True)
    return df

def calculate_metrics(df):
    """Calculates 30-day Volatility and Risk Score v2."""
    # Daily Returns
    df['Return'] = df['Close'].pct_change()
    
    # 1. Volatility (30-day Rolling Standard Deviation of Daily Returns)
    # NOT annualized, to match the 0-0.05 scale request.
    df['Vol_30d'] = df['Return'].rolling(window=30).std()
    
    # 2. Risk Score v2 (Vol-only fallback)
    # We need Annualized Volatility for the Risk Score formula to be consistent with other plots
    annualized_vol = df['Vol_30d'] * (252 ** 0.5)
    
    # Filter extreme glitches (> 100%)
    annualized_vol = annualized_vol.mask(annualized_vol > 1.0)
    
    # Z-score of volatility
    vol_mean = annualized_vol.mean()
    vol_std_dev = annualized_vol.std()
    if vol_std_dev == 0: vol_std_dev = 1e-6
    
    vol_z = (annualized_vol - vol_mean) / vol_std_dev
    
    # Risk Formula: 50 + (vol_z * 12)
    risk_score = 50 + (vol_z * 12.0)
    df['Risk_Score'] = risk_score.clip(0, 100)
    
    return df.dropna()

def plot_volatility_vs_risk(df):
    """Generates the dual-axis plot comparing Volatility and Risk Score."""
    # Visual Specifications
    plt.figure(figsize=(14, 7), dpi=300)
    
    # Create Dual Axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Colors
    color_vol = '#2563eb' # Blue
    color_risk = '#16a34a' # Green
    
    # --- Left Axis: Volatility ---
    ax1.plot(df.index, df['Vol_30d'], color=color_vol, linewidth=2, label='Vol 30d')
    ax1.set_ylabel('Volatility (30d)', fontsize=14, color=color_vol)
    ax1.tick_params(axis='y', labelcolor=color_vol)
    ax1.set_ylim(0, 0.045) # Requested scale 0 to 0.045
    
    # --- Right Axis: Risk Score ---
    ax2.plot(df.index, df['Risk_Score'], color=color_risk, linewidth=2, label='Risk Score v2')
    ax2.set_ylabel('Risk Score v2', fontsize=14, color=color_risk)
    ax2.tick_params(axis='y', labelcolor=color_risk)
    ax2.set_ylim(0, 100)
    
    # --- Annotations (Crisis Shading) ---
    # 2015-08 to 2015-10: Light red
    ax1.axvspan(pd.Timestamp('2015-08-01'), pd.Timestamp('2015-10-31'), color='red', alpha=0.2)
    
    # 2020-03 to 2020-04: Dark red
    ax1.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-04-30'), color='darkred', alpha=0.3)
    
    # 2022-01 to 2022-06: Light orange
    ax1.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-06-30'), color='orange', alpha=0.2)
    
    # Annotate Peaks
    # COVID Peak
    covid_data = df.loc['2020-03-01':'2020-04-30']
    if not covid_data.empty:
        max_vol_val = covid_data['Vol_30d'].max()
        max_vol_date = covid_data['Vol_30d'].idxmax()
        ax1.annotate('COVID Spike', 
                     xy=(max_vol_date, max_vol_val), 
                     xytext=(max_vol_date + pd.Timedelta(days=100), max_vol_val),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # --- Formatting ---
    plt.title("Volatility vs Composite Risk Score", fontsize=20, fontweight='bold', pad=20)
    ax1.set_xlabel("Date", fontsize=14)
    
    # Grid
    ax1.grid(True, which='major', alpha=0.2)
    
    # Date Format
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
    
    # Spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, shadow=True, fontsize=12)
    
    # Save
    output_file = 'vol_risk_composite_300dpi.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

def main():
    try:
        # 1. Fetch Data
        df = fetch_data()
        
        if df.empty:
            print("No data fetched.")
            return
            
        # 2. Calculate Metrics
        df = calculate_metrics(df)
        
        # 3. Visualize
        plot_volatility_vs_risk(df)
        print("Analysis complete.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
