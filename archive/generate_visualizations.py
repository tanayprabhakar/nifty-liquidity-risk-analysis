import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DATA_FILE = "master_market_data_2015_2022_final.csv"
OUTPUT_DIR = "."

# Plotting Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=1.0)
sns.set_palette("viridis")
from matplotlib import cm


def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

def main():
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Data loaded: {len(df)} rows.")

    # 1. NIFTY 50 Close Price
    print("Generating NIFTY 50 Close plot...")
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['NIFTY_50_Close'], color=cm.viridis(0.3), linewidth=2, label='NIFTY 50')
    plt.title('NIFTY 50 Close Price (2015–2025)', fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Index Level', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    save_plot('nifty_50_close_professional.png')

    # 2. Volatility vs Risk Score
    print("Generating Vol vs Risk plot...")
    fig, ax1 = plt.subplots(figsize=(14, 6))
    color = cm.viridis(0.3)
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Volatility (30d)', color=color, fontsize=14)
    ax1.plot(df['Date'], df['Vol_30d'], color=color, alpha=0.8, linewidth=1.5, label='Vol 30d')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = cm.viridis(0.8)
    ax2.set_ylabel('Risk Score v2', color=color, fontsize=14)
    ax2.plot(df['Date'], df['Risk_Score_v2'], color=color, alpha=0.8, linewidth=1.5, label='Risk Score v2')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(False)

    plt.title('Volatility vs Composite Risk Score', fontsize=18, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    save_plot('volatility_vs_risk_professional.png')

    # 3. Sector 30-Day Returns
    print("Generating Sector Momentum plot...")
    latest_row = df.dropna(subset=['NIFTY_50_Close']).iloc[-1]
    sector_cols = [c for c in df.columns if '30dRet' in c]
    if sector_cols:
        sector_data = latest_row[sector_cols].sort_values(ascending=False)
        sector_names = [s.replace('_30dRet', '').replace('NIFTY_', '') for s in sector_data.index]
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x=sector_data.values, y=sector_names, palette="viridis")
        plt.title(f'Sector 30-Day Momentum (Latest: {latest_row["Date"].date()})', fontsize=18, fontweight='bold')
        plt.xlabel('30-Day Return', fontsize=14)
        plt.ylabel('Sector', fontsize=14)
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_plot('sector_momentum_professional.png')

    # 4. Correlation Matrix
    print("Generating Correlation Matrix...")
    ret_cols = [c for c in df.columns if '_Return' in c]
    if ret_cols:
        corr = df[ret_cols].corr()
        clean_names = [c.replace('_Return', '').replace('NIFTY_', '') for c in corr.columns]
        corr.columns = clean_names
        corr.index = clean_names
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5, square=True)
        plt.title('Sector Return Correlation Matrix', fontsize=18, fontweight='bold')
        plt.tight_layout()
        save_plot('sector_correlation_professional.png')

    # 5. Rolling Beta
    print("Generating Rolling Beta plot...")
    target_sectors = ['NIFTY_IT', 'NIFTY_BANK', 'NIFTY_FMCG', 'NIFTY_AUTO', 'NIFTY_METAL']
    window = 30
    plt.figure(figsize=(14, 6))
    if 'NIFTY_50_Return' in df.columns:
        nifty_vol = df['NIFTY_50_Return'].rolling(window).std()
        for sector in target_sectors:
            col = f'{sector}_Return'
            if col in df.columns:
                sector_vol = df[col].rolling(window).std()
                corr = df[col].rolling(window).corr(df['NIFTY_50_Return'])
                beta = corr * (sector_vol / nifty_vol)
                clean_name = sector.replace('NIFTY_', '')
                plt.plot(df['Date'], beta, label=clean_name, linewidth=1.5)

        plt.title('30-Day Rolling Beta vs NIFTY 50', fontsize=18, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Beta', fontsize=14)
        plt.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Market Beta (1.0)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_plot('rolling_beta_professional.png')

    # 6. Market Regime Map
    print("Generating Market Regime Map...")
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['NIFTY_50_Close'], color='black', linewidth=1.5, label='NIFTY 50')
    y_min, y_max = plt.ylim()
    low_mask = df['Risk_Score_v2'] < 40
    med_mask = (df['Risk_Score_v2'] >= 40) & (df['Risk_Score_v2'] < 60)
    high_mask = df['Risk_Score_v2'] >= 60
    
    plt.fill_between(df['Date'], y_min, y_max, where=low_mask, color=cm.viridis(0.9), alpha=0.3, label='Low Risk (<40)')
    plt.fill_between(df['Date'], y_min, y_max, where=med_mask, color=cm.viridis(0.5), alpha=0.3, label='Medium Risk (40-60)')
    plt.fill_between(df['Date'], y_min, y_max, where=high_mask, color=cm.viridis(0.1), alpha=0.3, label='High Risk (>60)')
    
    plt.title('Market Risk Regimes (2015–2025)', fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('NIFTY 50 Level', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_plot('market_regime_map_professional.png')

    # 7. FII Lead-Lag
    print("Generating FII Lead-Lag plot...")
    lags = range(-10, 11)
    sectors_to_test = ['NIFTY_BANK', 'NIFTY_PSU_BANK', 'NIFTY_IT']
    plt.figure(figsize=(14, 6))
    if 'FII_Net' in df.columns:
        for sector in sectors_to_test:
            col = f'{sector}_Return'
            if col in df.columns:
                corrs = []
                for lag in lags:
                    c = df['FII_Net'].shift(lag).corr(df[col])
                    corrs.append(c)
                clean_name = sector.replace('NIFTY_', '')
                plt.plot(lags, corrs, marker='o', label=clean_name)
        
        plt.title('FII Flows Lead–Lag Impact on Sector Returns', fontsize=18, fontweight='bold')
        plt.xlabel('Lag (Days) [Positive = FII Leads]', fontsize=14)
        plt.ylabel('Correlation', fontsize=14)
        plt.axvline(0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(0, color='black', linestyle='-', alpha=0.2)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_plot('fii_lead_lag_professional.png')

if __name__ == "__main__":
    main()
