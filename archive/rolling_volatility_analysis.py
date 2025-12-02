import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_data():
    """Fetches 10 years of historical data for Nifty 50 and selected sectors."""
    tickers = {
        'Nifty 50': '^NSEI',
        'Nifty Bank': '^NSEBANK',
        'Nifty IT': '^CNXIT'
    }
    
    print("Fetching 10 years of data from yfinance...")
    # Fetch 10 years of data
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
    """Calculates annualized rolling volatility and market daily returns."""
    # Calculate daily percent change
    returns = df.pct_change().dropna()
    
    # --- Market Daily Returns (Context) ---
    market_col = 'Nifty 50'
    market_returns = returns[market_col]
    # We use raw daily returns as requested to see every spike
    
    # --- Rolling Volatility (Risk) ---
    volatility = pd.DataFrame(index=returns.index)
    
    for col in ['Nifty Bank', 'Nifty IT']:
        if col not in returns.columns:
            continue
            
        # Calculate 30-Day Rolling Standard Deviation
        rolling_std = returns[col].rolling(window=window).std()
        
        # Annualize it: volatility = rolling_std * (252 ** 0.5)
        annualized_vol = rolling_std * (252 ** 0.5)
        
        # Filter out extreme data glitches (volatility > 100% i.e., > 1.0)
        annualized_vol = annualized_vol.mask(annualized_vol > 1.0)
        
        volatility[col] = annualized_vol * 100 # Convert to percentage for plotting
        
    return volatility, market_returns # Return raw daily returns (decimal)

def plot_volatility(volatility, market_returns):
    """Plots the rolling volatility with Market Daily Returns background."""
    # Use seaborn-whitegrid theme
    sns.set_theme(style="whitegrid")
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # --- Ax2 (Right Axis): Market Daily Returns ---
    ax2 = ax1.twinx()
    
    # Plot Market Daily Returns as area chart in background
    # zorder=0 ensures it stays behind the volatility lines
    # Color RED (alpha=0.3) where daily_return < 0 (Loss/Crash)
    # Color GREEN (alpha=0.3) where daily_return > 0 (Gain/Rally)
    
    ax2.fill_between(market_returns.index, market_returns, 0, 
                     where=(market_returns >= 0), color='green', alpha=0.3, label='Market Rally', zorder=0)
    ax2.fill_between(market_returns.index, market_returns, 0, 
                     where=(market_returns < 0), color='red', alpha=0.3, label='Market Crash', zorder=0)
    
    # Set the Right Y-Axis limits to -0.10 to +0.10 (-10% to +10%)
    ax2.set_ylim(-0.10, 0.10)
    
    # --- Ax1 (Left Axis): Annualized Volatility ---
    # zorder=10 ensures they stay on top
    for col in volatility.columns:
        ax1.plot(volatility.index, volatility[col], label=f'{col} Volatility', linewidth=2, zorder=10)
        
    # --- Formatting ---
    ax1.set_title('Sector Risk vs Market Crashes (Daily)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Annualized Volatility (%)')
    ax2.set_ylabel('Market Daily Return')
    
    # Combine legends or show primary
    ax1.legend(loc='upper left')
    
    plt.tight_layout()
    
    output_file = 'rolling_volatility_output.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()

def main():
    try:
        # 1. Data Setup
        df = fetch_data()
        
        if df.empty:
            print("No data fetched. Please check your internet connection or ticker symbols.")
            return

        # 2. & 3. Metrics Calculation
        volatility, market_returns = calculate_metrics(df)
        
        # 4. Visualization
        plot_volatility(volatility, market_returns)
        print("Analysis complete.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
