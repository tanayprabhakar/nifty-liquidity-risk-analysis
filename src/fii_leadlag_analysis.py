import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_stuff(path):
    print(f"loading {path}...")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # check cols
    req = ['FII_Net', 'NIFTY_BANK_Return', 'NIFTY_PSU_BANK_Return', 'NIFTY_IT_Return']
    miss = [c for c in req if c not in df.columns]
    
    if miss:
        print(f"missing {miss}, trying to calc...")
        for c in ['NIFTY_BANK', 'NIFTY_PSU_BANK', 'NIFTY_IT']:
            cc = f"{c}_Close"
            rc = f"{c}_Return"
            if rc in miss and cc in df.columns:
                df[rc] = df[cc].pct_change()
    
    return df

def get_lead_lag(df, lags=range(-10, 11)):
    # sectors to check
    secs = {
        'BANK': 'NIFTY_BANK_Return',
        'PSU_BANK': 'NIFTY_PSU_BANK_Return',
        'IT': 'NIFTY_IT_Return'
    }
    
    res = {s: [] for s in secs}
    lag_list = list(lags)
    
    for l in lag_list:
        # shift fii
        fii_s = df['FII_Net'].shift(l)
        
        for s, r in secs.items():
            c = fii_s.corr(df[r])
            res[s].append(c)
            
    return lag_list, res

def draw_plot(lags, corrs):
    plt.figure(figsize=(12, 7), dpi=300)
    
    # styles
    st = {
        'BANK': {'c': '#1e3a8a', 'ls': '-', 'lw': 2.5, 'lbl': 'BANK', 'm': 'o'},
        'PSU_BANK': {'c': '#0369a1', 'ls': '--', 'lw': 2.5, 'lbl': 'PSU_BANK', 'm': 's'},
        'IT': {'c': '#0d9488', 'ls': ':', 'lw': 2.5, 'lbl': 'IT', 'm': '^'}
    }
    
    for s, vals in corrs.items():
        sty = st[s]
        # find peak
        pidx = np.argmax(vals)
        plag = lags[pidx]
        pval = vals[pidx]
        
        plt.plot(lags, vals, color=sty['c'], linestyle=sty['ls'], linewidth=sty['lw'], label=sty['lbl'])
        plt.plot(plag, pval, marker=sty['m'], markersize=8, color=sty['c'])

    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.title("FII Flows Lead-Lag Impact on Sector Returns", fontsize=18, fontweight='bold')
    plt.xlabel("Lag (Days) [Positive = FII Leads]", fontsize=14)
    plt.ylabel("Correlation", fontsize=14)
    plt.ylim(-0.05, 0.20)
    plt.xlim(-10, 10)
    plt.legend()
    
    # annotations
    b_vals = corrs['BANK']
    z_idx = lags.index(0)
    b_0 = b_vals[z_idx]
    
    plt.annotate('Same-day peak', xy=(0, b_0), xytext=(2, b_0 + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
                 
    plt.text(5, 0.18, "Positive Lag = FII Leads", fontsize=12, style='italic', 
             bbox=dict(facecolor='white', alpha=0.8))

    out = '../plots/fii_leadlag_impact_300dpi.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"saved {out}")
    plt.close()

def main():
    try:
        f = "../data/master_market_data_2015_2022_final.csv"
        df = load_stuff(f)
        l, c = get_lead_lag(df)
        draw_plot(l, c)
        print("done.")
    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    main()
