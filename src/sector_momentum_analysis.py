import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick

def get_data():
    ticks = {
        'NIFTY AUTO': '^CNXAUTO',
        'NIFTY BANK': '^NSEBANK',
        'NIFTY ENERGY': '^CNXENERGY',
        'NIFTY FMCG': '^CNXFMCG',
        'NIFTY IT': '^CNXIT',
        'NIFTY METAL': '^CNXMETAL',
        'NIFTY PHARMA': '^CNXPHARMA',
        'NIFTY PSU BANK': '^CNXPSUBANK',
        'NIFTY REALTY': '^CNXREALTY',
        'NIFTY INFRA': '^CNXINFRA',
        'NIFTY 50': '^NSEI'
    }
    
    print("fetching data...")
    try:
        df = yf.download(list(ticks.values()), start='2025-09-01', end='2025-11-20', progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df['Adj Close']
            except:
                df = df['Close']
                
        inv = {v: k for k, v in ticks.items()}
        df.rename(columns=inv, inplace=True)
        return df
    except:
        return pd.DataFrame()

def get_mom(df):
    t_date = pd.Timestamp('2025-11-18')
    s_date = t_date - pd.Timedelta(days=30)
    
    try:
        e_idx = df.index.get_indexer([t_date], method='nearest')[0]
        s_idx = df.index.get_indexer([s_date], method='nearest')[0]
        
        e_p = df.iloc[e_idx]
        s_p = df.iloc[s_idx]
        
        ret = (e_p - s_p) / s_p
        return ret
    except:
        return pd.Series()

def draw_plot(rets):
    nifty = rets['NIFTY 50']
    secs = rets.drop('NIFTY 50').sort_values(ascending=True)
    
    plt.figure(figsize=(10, 8), dpi=300)
    
    # colors
    cmap = LinearSegmentedColormap.from_list("custom", ["#5eead4", "#6b21a8"])
    n = len(secs)
    cols = [cmap(i/(n-1)) for i in range(n)]
    
    bars = plt.barh(secs.index, secs, color=cols, height=0.75)
    
    for b in bars:
        w = b.get_width()
        x = w if w > 0 else w - 0.01
        ha = 'left' if w > 0 else 'right'
        
        plt.text(x + (0.002 if w > 0 else -0.002), 
                 b.get_y() + b.get_height()/2, 
                 f'{w:.1%}', 
                 va='center', ha=ha, fontsize=11, fontweight='bold', color='black')
                 
    plt.axvline(x=nifty, color='gray', linestyle='--', alpha=0.7)
    
    # arrow
    top = bars[-1]
    plt.annotate('Leading Sector', 
                 xy=(top.get_width(), top.get_y() + top.get_height()/2), 
                 xytext=(top.get_width() + 0.02, top.get_y() + top.get_height()/2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, fontweight='bold', va='center')
                 
    plt.figtext(0.5, 0.02, "Data as of November 18, 2025", ha="center", fontsize=10, style='italic')
    
    plt.title("Sector 30-Day Momentum (Latest: 2025-11-18)", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("30-Day Return", fontsize=14)
    plt.grid(True, axis='x', alpha=0.25)
    
    ax = plt.gca()
    ax.set_facecolor('#f8fafc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    out = '../plots/sector_momentum_30d_300dpi.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"saved {out}")
    plt.close()

def main():
    try:
        df = get_data()
        if df.empty: return
        rets = get_mom(df)
        if rets.empty: return
        draw_plot(rets)
        print("done.")
    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    main()
