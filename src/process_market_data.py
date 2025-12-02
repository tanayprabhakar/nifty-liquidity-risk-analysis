import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# settings
data_dir = "../data" 
out_dir = "../data" # csvs go to data
plot_dir = "../plots"
fii_file = "FiiDiiTradingactivity.csv"
clean_fii = "FII_DII_cleaned.csv"
master_file = "master_market_data_2015_2022.csv"
final_file = "master_market_data_2015_2022_final.csv"

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

# 1. clean up the flow data
def fix_flow_data(path):
    print(f"cleaning {path}...")
    try:
        df = pd.read_csv(path)
    except:
        print("file not found.")
        return None

    # fix dates
    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    except:
        pass

    df = df.dropna(subset=['Date'])

    # helper to fix numbers
    def fix_num(x):
        if isinstance(x, str):
            x = x.replace(',', '').replace('(', '-').replace(')', '').replace(' ', '')
        return pd.to_numeric(x, errors='coerce')

    # fix columns
    df.columns = [c.strip() for c in df.columns]
    
    for c in df.columns:
        if c != 'Date':
            df[c] = df[c].apply(fix_num)

    # try to find cols
    fii_buy = next((c for c in df.columns if 'FII' in c and 'Buy' in c), None)
    fii_sell = next((c for c in df.columns if 'FII' in c and 'Sell' in c), None)
    fii_net = next((c for c in df.columns if 'FII' in c and 'Net' in c), None)
    
    dii_buy = next((c for c in df.columns if 'DII' in c and 'Buy' in c), None)
    dii_sell = next((c for c in df.columns if 'DII' in c and 'Sell' in c), None)
    dii_net = next((c for c in df.columns if 'DII' in c and 'Net' in c), None)

    out = pd.DataFrame()
    out['Date'] = df['Date']
    
    if fii_buy: out['FII_Buy'] = df[fii_buy]
    if fii_sell: out['FII_Sell'] = df[fii_sell]
    if fii_net: out['FII_Net'] = df[fii_net]
    
    if dii_buy: out['DII_Buy'] = df[dii_buy]
    if dii_sell: out['DII_Sell'] = df[dii_sell]
    if dii_net: out['DII_Net'] = df[dii_net]

    # calc net if missing
    if 'FII_Net' not in out.columns and 'FII_Buy' in out and 'FII_Sell' in out:
        out['FII_Net'] = out['FII_Buy'] - out['FII_Sell']
    
    if 'DII_Net' not in out.columns and 'DII_Buy' in out and 'DII_Sell' in out:
        out['DII_Net'] = out['DII_Buy'] - out['DII_Sell']

    out = out.sort_values('Date')
    out.to_csv(os.path.join(out_dir, clean_fii), index=False)
    print(f"saved {clean_fii}")
    return out

# 2. read sector files
def read_sector_files():
    files = glob.glob(os.path.join(data_dir, "NIFTY_*.csv"))
    data = {}
    
    print(f"found {len(files)} files.")
    
    for f in files:
        name = os.path.basename(f).replace('.csv', '')
        print(f"reading {name}...")
        try:
            df = pd.read_csv(f)
            # fix date
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            else:
                d_col = next((c for c in df.columns if 'Date' in c or 'date' in c), None)
                if d_col:
                    df = df.rename(columns={d_col: 'Date'})
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date')
            
            # get close price
            c_col = next((c for c in df.columns if 'Close' in c), None)
            if c_col:
                df = df[['Date', c_col]].rename(columns={c_col: f'{name}_Close'})
                df[f'{name}_Close'] = pd.to_numeric(df[f'{name}_Close'], errors='coerce')
                data[name] = df
            else:
                print(f"no close col in {name}")
                
        except Exception as e:
            print(f"error {name}: {e}")
            
    return data

# 3. make master df
def make_master_df():
    fii_data = fix_flow_data(os.path.join(data_dir, fii_file))
    sec_data = read_sector_files()
    
    if 'NIFTY_50' not in sec_data:
        print("missing nifty 50.")
        return None

    master = sec_data['NIFTY_50'].copy()
    
    for name, d in sec_data.items():
        if name == 'NIFTY_50': continue
        master = pd.merge(master, d, on='Date', how='left')
        
    if fii_data is not None:
        master = pd.merge(master, fii_data, on='Date', how='left')
        
    master.to_csv(os.path.join(out_dir, master_file), index=False)
    print(f"saved {master_file}")
    
    return master

# 4. add calcs
def add_calcs(df):
    print("adding features...")
    df = df.copy()
    df = df.sort_values('Date')
    
    # returns
    cols = [c for c in df.columns if 'Close' in c]
    for c in cols:
        ret = c.replace('Close', 'Return')
        df[ret] = df[c].pct_change()
        
        # 30d
        ret30 = c.replace('Close', '30dRet')
        df[ret30] = df[c].pct_change(periods=30)

    # volatility
    if 'NIFTY_50_Return' in df.columns:
        df['Vol_7d'] = df['NIFTY_50_Return'].rolling(7).std()
        df['Vol_30d'] = df['NIFTY_50_Return'].rolling(30).std()
        df['Vol_90d'] = df['NIFTY_50_Return'].rolling(90).std()
        
    return df

# 5. calc risk score
def calc_risk_v2(df):
    print("calculating risk...")
    df = df.copy()
    
    w_vol = 12.0
    w_fii = 6.0
    
    # z-score vol
    if 'Vol_30d' in df.columns:
        v = df['Vol_30d'].dropna()
        df['vol_z'] = (df['Vol_30d'] - v.mean()) / (v.std() if v.std() != 0 else 1e-6)
    else:
        df['vol_z'] = 0
        
    # z-score fii
    if 'FII_Net' in df.columns:
        f = df['FII_Net'].dropna()
        df['fii_z'] = (df['FII_Net'] - f.mean()) / (f.std() if f.std() != 0 else 1e-6)
    else:
        df['fii_z'] = 0
        
    # formula
    df['Risk_Score_v2'] = 50 + (df['vol_z'] * w_vol) - (df['fii_z'] * w_fii)
    df['Risk_Score_v2'] = df['Risk_Score_v2'].clip(0, 100)
    
    return df

# 6. make charts
def make_charts(df):
    print("making plots...")
    
    # nifty close
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['NIFTY_50_Close'], label='NIFTY 50')
    plt.title('NIFTY 50 Close Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_nifty_close.png'), dpi=300)
    plt.close()
    
    # vol vs risk
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(df['Date'], df['Vol_30d'], color='blue', alpha=0.6, label='Vol 30d')
    ax2.plot(df['Date'], df['Risk_Score_v2'], color='red', alpha=0.6, label='Risk Score v2')
    
    ax1.set_ylabel('Volatility (30d)', color='blue')
    ax2.set_ylabel('Risk Score', color='red')
    plt.title('Volatility vs Risk Score')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_vol_risk.png'), dpi=300)
    plt.close()
    
    # sector 30d
    last = df.dropna(subset=['NIFTY_50_Close']).iloc[-1]
    s_cols = [c for c in df.columns if '30dRet' in c]
    if s_cols:
        l_rets = last[s_cols].sort_values()
        plt.figure(figsize=(12, 8))
        sns.barplot(x=l_rets.values, y=l_rets.index)
        plt.title(f'Sector 30d Returns ({last["Date"].date()})')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'plot_sector_30d.png'), dpi=300)
        plt.close()
        
    # corr matrix
    r_cols = [c for c in df.columns if '_Return' in c]
    if r_cols:
        c = df[r_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(c, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Sector Return Correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'plot_corr_matrix.png'), dpi=300)
        plt.close()

def main():
    # run it all
    data = make_master_df()
    if data is None: return
    
    data = add_calcs(data)
    data = calc_risk_v2(data)
    
    # diagnostics
    print("\ndiagnostics:")
    print(f"rows: {len(data)}")
    
    # rolling corr
    if 'NIFTY_50_Return' in data.columns:
        print("\ncalc rolling corr...")
        for c in data.columns:
            if '_Return' in c and c != 'NIFTY_50_Return':
                s = c.replace('_Return', '')
                data[f'{s}_Corr_30d'] = data[c].rolling(30).corr(data['NIFTY_50_Return'])

    # drawdown
    if 'NIFTY_50_Close' in data.columns:
        rmax = data['NIFTY_50_Close'].cummax()
        dd = (data['NIFTY_50_Close'] - rmax) / rmax
        print(f"\nmax dd: {dd.min():.2%} on {data.loc[dd.idxmin(), 'Date'].date()}")
        
    # regimes
    data['Risk_Regime'] = pd.cut(data['Risk_Score_v2'], 
                               bins=[-np.inf, 40, 60, np.inf], 
                               labels=['Low', 'Medium', 'High'])
    
    data.to_csv(os.path.join(out_dir, final_file), index=False)
    print(f"\nsaved {final_file}")
    
    make_charts(data)
    print("done.")

if __name__ == "__main__":
    main()
