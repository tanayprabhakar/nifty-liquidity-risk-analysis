import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def fetch_data():
    """Fetches data for Nifty Bank, IT, and Nifty 50."""
    tickers = {
        'BANK': '^NSEBANK',
        'IT': '^CNXIT',
        'NIFTY 50': '^NSEI'
    }
    
    print("Fetching data from 2017-01-01 to 2026-11-28...")
    data = yf.download(list(tickers.values()), start='2017-01-01', end='2026-11-28', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data['Adj Close']
        except KeyError:
            data = data['Close']
            
    # Rename columns
    inv_tickers = {v: k for k, v in tickers.items()}
    data.rename(columns=inv_tickers, inplace=True)
    
    # Drop rows with missing values
    data.dropna(inplace=True)
    return data

def calculate_metrics(df):
    """Calculates Annualized Volatility and Market Daily Returns."""
    returns = df.pct_change().dropna()
    
    # 1. Market Daily Return
    market_returns = returns['NIFTY 50']
    
    # 2. Annualized Volatility (30-day Rolling Std * sqrt(252))
    volatility = pd.DataFrame(index=returns.index)
    for col in ['BANK', 'IT']:
        rolling_std = returns[col].rolling(window=30).std()
        vol = rolling_std * (252 ** 0.5) * 100 # Convert to %
        # Filter extreme glitches (> 100%)
        vol = vol.mask(vol > 100)
        volatility[col] = vol
        
    return volatility.dropna(), market_returns

def plot_risk_crashes(volatility, market_returns):
    """Generates the dual-axis plot comparing Sector Risk vs Market Crashes."""
    # Visual Specifications
    plt.figure(figsize=(14, 6), dpi=300)
    
    # Create Dual Axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # --- Right Axis (Background): Market Daily Return ---
    # Plot bars
    # Using bar chart with width=1.5 for visibility on daily data over 9 years
    # Or use fill_between/vlines. Bar is requested.
    # Since we have ~2500 points, bars might be too dense. Let's try bar with width=1.0
    # But matplotlib bar is slow for many points. 'vlines' or 'fill_between' is better.
    # User asked for "Light green bars... light red bars".
    # Let's use fill_between which looks like bars when dense.
    
    ax2.fill_between(market_returns.index, 0, market_returns, where=(market_returns >= 0), 
                     color='lightgreen', alpha=0.6, label='Market Return (+)', step='mid')
    ax2.fill_between(market_returns.index, 0, market_returns, where=(market_returns < 0), 
                     color='lightcoral', alpha=0.6, label='Market Return (-)', step='mid')
                     
    ax2.set_ylabel('Market Daily Return', fontsize=14)
    ax2.set_ylim(-0.10, 0.10)
    
    # --- Left Axis (Foreground): Annualized Volatility ---
    # BANK: Navy (#1e40af)
    ax1.plot(volatility.index, volatility['BANK'], color='#1e40af', linewidth=1.5, label='Nifty Bank Volatility')
    # IT: Orange (#f97316)
    ax1.plot(volatility.index, volatility['IT'], color='#f97316', linewidth=1.5, label='Nifty IT Volatility')
    
    ax1.set_ylabel('Annualized Volatility (%)', fontsize=14)
    ax1.set_ylim(0, 100)
    
    # Ensure Volatility is on top
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    
    # --- Annotations ---
    # COVID Peak (March 2020)
    covid_start = pd.Timestamp('2020-02-20')
    covid_end = pd.Timestamp('2020-04-30')
    ax1.axvspan(covid_start, covid_end, color='red', alpha=0.15)
    
    # Peak Volatility Annotation
    # Find max vol for BANK
    max_vol_bank = volatility['BANK'].max()
    max_vol_date = volatility['BANK'].idxmax()
    
    ax1.annotate(f'Peak Vol: {max_vol_bank:.1f}%', 
                 xy=(max_vol_date, max_vol_bank), 
                 xytext=(max_vol_date + pd.Timedelta(days=100), max_vol_bank),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, fontweight='bold')

    # --- Formatting ---
    plt.title("Sector Risk vs Market Crashes (Daily)", fontsize=18, fontweight='bold', pad=20)
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
    # We don't necessarily need legend for the background bars if not requested, 
    # but user asked for "Nifty Bank Volatility", "Nifty IT Volatility".
    ax1.legend(lines1, labels1, loc='upper left', frameon=True, fontsize=11)
    
    # Save
    output_file = 'sector_risk_crashes_300dpi.png'
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
        volatility, market_returns = calculate_metrics(df)
        
        # 3. Visualize
        plot_risk_crashes(volatility, market_returns)
        print("Analysis complete.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
