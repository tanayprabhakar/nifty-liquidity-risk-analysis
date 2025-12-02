import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def get_nifty_data():
    print("getting nifty data...")
    try:
        data = yf.download('^NSEI', start='2015-01-01', end='2022-12-31', progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            try:
                close = data['Adj Close']
            except:
                close = data['Close']
        else:
            close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            
        df = pd.DataFrame({'Close': close})
        df.dropna(inplace=True)
        return df
    except:
        return pd.DataFrame()

def find_regimes(df):
    # returns
    df['Return'] = df['Close'].pct_change()
    
    # vol
    roll = df['Return'].rolling(30).std()
    ann_vol = roll * (252 ** 0.5)
    ann_vol = ann_vol.mask(ann_vol > 1.0)
    
    # risk score
    v_mean = ann_vol.mean()
    v_std = ann_vol.std()
    if v_std == 0: v_std = 1e-6
    
    z = (ann_vol - v_mean) / v_std
    
    score = 50 + (z * 12.0)
    df['Risk_Score'] = score.clip(0, 100)
    
    return df.dropna()

def draw_plot(df):
    plt.figure(figsize=(14, 7), dpi=300)
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # nifty
    ax1.plot(df.index, df['Close'], color='black', linewidth=2, label='NIFTY 50')
    ax1.set_ylabel('NIFTY 50 Level', fontsize=14)
    ax1.set_ylim(6000, 20000)
    
    # risk
    ax2.plot(df.index, df['Risk_Score'], color='none', alpha=0) 
    ax2.set_ylabel('Risk Score v2', fontsize=14)
    ax2.set_ylim(0, 100)
    
    # bands
    ax2.fill_between(df.index, 0, 100, where=(df['Risk_Score'] < 40), 
                     color='#4ade80', alpha=0.4, label='Low Risk')
    
    ax2.fill_between(df.index, 0, 100, where=((df['Risk_Score'] >= 40) & (df['Risk_Score'] <= 60)), 
                     color='#facc15', alpha=0.4, label='Medium Risk')
                     
    ax2.fill_between(df.index, 0, 100, where=(df['Risk_Score'] > 60), 
                     color='#f87171', alpha=0.4, label='High Risk')
    
    # events
    evs = [
        ('2015-08-24', '2015 China Slowdown', 'red'),
        ('2018-09-01', '2018 IL&FS Crisis', 'orange'),
        ('2020-03-23', '2020 COVID Crash', 'darkred'),
        ('2022-06-15', '2022 Global Tightening', 'purple')
    ]
    
    for d, l, c in evs:
        do = pd.to_datetime(d)
        if do >= df.index.min() and do <= df.index.max():
            ax1.axvline(x=do, color=c, linestyle='--', alpha=0.8)
            ax1.text(do, ax1.get_ylim()[1]*0.95, f' {l}', 
                     color=c, rotation=90, va='top', fontsize=10, fontweight='bold')

    plt.title("Market Risk Regimes (2015-2022)", fontsize=20, fontweight='bold', pad=20)
    ax1.set_xlabel("Date", fontsize=14)
    ax1.grid(True, alpha=0.25)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    
    # legend
    legs = [
        Line2D([0], [0], color='black', lw=2, label='NIFTY 50'),
        Patch(facecolor='#4ade80', alpha=0.4, label='Low Risk'),
        Patch(facecolor='#facc15', alpha=0.4, label='Medium Risk'),
        Patch(facecolor='#f87171', alpha=0.4, label='High Risk')
    ]
    
    ax1.legend(handles=legs, loc='upper left', frameon=True, shadow=True)
    
    out = '../plots/market_risk_regimes_300dpi.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"saved {out}")
    plt.close()

def main():
    try:
        df = get_nifty_data()
        if df.empty: return
        df = find_regimes(df)
        draw_plot(df)
        print("done.")
    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    main()
