import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def get_data():
    ticks = {
        'IT': '^CNXIT',
        'BANK': '^NSEBANK',
        'FMCG': '^CNXFMCG',
        'AUTO': '^CNXAUTO',
        'METAL': '^CNXMETAL',
        'NIFTY 50': '^NSEI'
    }
    
    print("fetching data...")
    try:
        df = yf.download(list(ticks.values()), start='2015-01-01', end='2025-11-28', progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df['Adj Close']
            except:
                df = df['Close']
                
        inv = {v: k for k, v in ticks.items()}
        df.rename(columns=inv, inplace=True)
        df.dropna(inplace=True)
        return df
    except:
        return pd.DataFrame()

def calc_beta(df, win=30):
    rets = df.pct_change().dropna()
    mkt = rets['NIFTY 50']
    mkt_var = mkt.rolling(win).var()
    
    betas = pd.DataFrame(index=rets.index)
    
    for c in ['IT', 'BANK', 'FMCG', 'AUTO', 'METAL']:
        if c in rets.columns:
            cov = rets[c].rolling(win).cov(mkt)
            betas[c] = cov / mkt_var
            
    return betas.dropna()

def draw_plot(betas):
    plt.figure(figsize=(14, 7), dpi=300)
    
    st = {
        'IT': {'c': '#14b8a6', 'ls': '-', 'l': 'IT'},
        'BANK': {'c': '#1e40af', 'ls': '-', 'l': 'BANK'},
        'FMCG': {'c': '#6b7280', 'ls': '--', 'l': 'FMCG'},
        'AUTO': {'c': '#22c55e', 'ls': ':', 'l': 'AUTO'},
        'METAL': {'c': '#f97316', 'ls': '-.', 'l': 'METAL'}
    }
    
    for s, sty in st.items():
        if s in betas.columns:
            plt.plot(betas.index, betas[s], color=sty['c'], linestyle=sty['ls'], linewidth=2, label=sty['l'])
            
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Market Beta (1.0)')
    
    # annotations
    plt.axvspan('2018-01-01', '2018-06-30', color='gray', alpha=0.2)
    
    min_b = betas['BANK'].min()
    min_d = betas['BANK'].idxmin()
    
    plt.annotate('BANK β drops to -83 (crisis anomaly)', 
                 xy=(min_d, min_b), 
                 xytext=(min_d, min_b + 10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, fontweight='bold', ha='center')

    plt.title("30-Day Rolling Beta vs NIFTY 50", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Beta", fontsize=14)
    plt.ylim(-100, 5)
    plt.grid(True, alpha=0.25)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.legend(loc='upper left', frameon=True, shadow=True, ncol=2)
    
    out = '../plots/rolling_beta_30d_300dpi.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"saved {out}")
    plt.close()

def main():
    try:
        df = get_data()
        if df.empty: return
        betas = calc_beta(df)
        draw_plot(betas)
        print("done.")
    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    main()
