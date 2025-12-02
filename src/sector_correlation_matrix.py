import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_data():
    ticks = {
        'NIFTY 50': '^NSEI',
        'AUTO': '^CNXAUTO',
        'BANK': '^NSEBANK',
        'ENERGY': '^CNXENERGY',
        'FMCG': '^CNXFMCG',
        'IT': '^CNXIT',
        'METAL': '^CNXMETAL',
        'PHARMA': '^CNXPHARMA',
        'PSU BANK': '^CNXPSUBANK',
        'REALTY': '^CNXREALTY'
    }
    
    print("fetching data...")
    try:
        df = yf.download(list(ticks.values()), period="2y", progress=False)
        
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

def get_corr(df):
    rets = df.pct_change().dropna()
    c = rets.corr()
    return c

def draw_plot(c):
    plt.figure(figsize=(10, 10), dpi=300)
    
    mask = np.triu(np.ones_like(c, dtype=bool))
    
    sns.heatmap(c, mask=mask, cmap='RdYlGn', center=0,
                annot=True, fmt=".2f", annot_kws={"size": 11},
                linewidths=0.5, linecolor='lightgray',
                cbar_kws={"label": "Correlation", "shrink": 0.8})
    
    plt.title("Sector Return Correlation Matrix", fontsize=20, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold')
    plt.yticks(rotation=0, fontsize=11, fontweight='bold')
    
    out = '../plots/sector_correlation_matrix_300dpi.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"saved {out}")
    plt.close()

def main():
    try:
        df = get_data()
        if df.empty: return
        c = get_corr(df)
        draw_plot(c)
        print("done.")
    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    main()
