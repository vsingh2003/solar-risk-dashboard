import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings, os
warnings.filterwarnings('ignore')
os.makedirs('images', exist_ok=True)

print("\n" + "="*65)
print("  PHASE 1: LOADING, CLEANING & MERGING DATASETS")
print("="*65)

print("\n📂 Loading Visser_plant_report.csv ...")
visser_raw = pd.read_csv('Visser_plant_report.csv')

visser_raw.columns = visser_raw.columns.str.strip()

print(f"   Raw shape  : {visser_raw.shape}")
print(f"   Duplicates : {visser_raw.duplicated().sum()} rows removed")
visser_raw = visser_raw.drop_duplicates()

owner_nulls = visser_raw['Owner'].isnull().sum()
print(f"   'Owner' col: {owner_nulls}/12 values are NaN → column dropped")
visser_raw = visser_raw.drop(columns=['Owner'])

visser_raw['earning_cad'] = (
    visser_raw['This Month Earning']
    .str.replace('CAD', '', regex=False)
    .str.strip()
    .astype(float)
)
print(f"   'This Month Earning' cleaned → 'earning_cad' (float, CAD)")

status_vals = visser_raw['Current Status'].unique()
print(f"   'Current Status' values: {status_vals} "
      f"— plant is offline; kept as reference flag")

def parse_visser_month(date_str):
    try:
        yy, mon = date_str.strip().split('-')
        return pd.to_datetime(f"01-{mon}-20{yy}", format='%d-%b-%Y')
    except Exception:
        return pd.NaT

visser_raw['month_parsed'] = visser_raw['Date'].apply(parse_visser_month)
bad_dates = visser_raw['month_parsed'].isna().sum()
if bad_dates:
    print(f"   ⚠ {bad_dates} unparseable dates dropped")
    visser_raw = visser_raw.dropna(subset=['month_parsed'])

day_cols = [str(i).zfill(2) for i in range(1, 32)]
day_cols  = [c for c in day_cols if c in visser_raw.columns]

visser_long = visser_raw.melt(
    id_vars=['month_parsed', 'Inverter Number', 'Current Status',
             'Installed Capacity (kWp)', 'Monthly Yield (kWh)',
             'This Month Full Load Hours (h)', 'earning_cad'],
    value_vars=day_cols,
    var_name='day_num',
    value_name='daily_yield_kwh'
)

visser_long['date'] = pd.to_datetime(
    visser_long['month_parsed'].dt.year.astype(str) + '-' +
    visser_long['month_parsed'].dt.month.astype(str).str.zfill(2) + '-' +
    visser_long['day_num'],
    errors='coerce'
)

nan_yield  = visser_long['daily_yield_kwh'].isna().sum()
nan_date   = visser_long['date'].isna().sum()
print(f"   Day NaN yields   : {nan_yield} rows dropped "
      f"(months shorter than 31 days)")
print(f"   Invalid calendar : {nan_date} rows dropped "
      f"(e.g. Feb-30, Apr-31)")

visser_long = visser_long.dropna(subset=['date', 'daily_yield_kwh'])

zero_yield = (visser_long['daily_yield_kwh'] == 0).sum()
print(f"   Zero-yield days  : {zero_yield} rows removed "
      f"(no production recorded)")
visser_long = visser_long[visser_long['daily_yield_kwh'] > 0].copy()

neg_yield = (visser_long['daily_yield_kwh'] < 0).sum()
if neg_yield:
    print(f"   Negative yields  : {neg_yield} rows removed")
    visser_long = visser_long[visser_long['daily_yield_kwh'] >= 0]

visser_long['source']          = 'Visser'
visser_long['unit_id']         = ('Visser-INV-' +
                                   visser_long['Inverter Number'].astype(str))
visser_long['installed_kw']    = visser_long['Installed Capacity (kWp)']
visser_long['full_load_hours'] = visser_long['This Month Full Load Hours (h)']
visser_long['monthly_yield']   = visser_long['Monthly Yield (kWh)']
visser_long['plant_status']    = visser_long['Current Status']
visser_long['earning_cad']     = visser_long['earning_cad']

visser_df = visser_long[[
    'date', 'unit_id', 'source', 'daily_yield_kwh',
    'installed_kw', 'full_load_hours', 'monthly_yield',
    'plant_status', 'earning_cad'
]].copy()

print(f"\n   ✅ Visser clean : {len(visser_df):>4} daily records "
      f"| Units : {visser_df['unit_id'].unique().tolist()}")

print("\n📂 Loading Bissell_Thrift_118_Ave_01012025-01012026 (1).xlsx ...")
bissell_raw = pd.read_excel(
    'Bissell_Thrift_118_Ave_01012025-01012026 (1).xlsx',
    skiprows=1
)
bissell_raw.columns = [
    'date',
    'inv1_kwh', 'inv2_kwh', 'inv3_kwh',
    'inv1_kwh_per_kwp', 'inv2_kwh_per_kwp', 'inv3_kwh_per_kwp',
    'total_kwh'
]
print(f"   Raw shape  : {bissell_raw.shape}")

bissell_raw['date'] = pd.to_datetime(
    bissell_raw['date'], format='%d.%m.%Y', errors='coerce'
)
bad_dates_b = bissell_raw['date'].isna().sum()
if bad_dates_b:
    print(f"   ⚠ {bad_dates_b} unparseable dates dropped")
bissell_raw = bissell_raw.dropna(subset=['date'])

energy_cols = [
    'inv1_kwh', 'inv2_kwh', 'inv3_kwh',
    'inv1_kwh_per_kwp', 'inv2_kwh_per_kwp', 'inv3_kwh_per_kwp',
    'total_kwh'
]
for col in energy_cols:
    bissell_raw[col] = pd.to_numeric(bissell_raw[col], errors='coerce').fillna(0)

dups = bissell_raw.duplicated(subset=['date']).sum()
print(f"   Duplicates : {dups} rows removed")
bissell_raw = bissell_raw.drop_duplicates(subset=['date'])

for col in ['inv1_kwh', 'inv2_kwh', 'inv3_kwh']:
    neg = (bissell_raw[col] < 0).sum()
    if neg:
        print(f"   ⚠ Negative {col}: {neg} values set to 0")
        bissell_raw[col] = bissell_raw[col].clip(lower=0)

bissell_raw['inv_sum'] = (
    bissell_raw['inv1_kwh'] + bissell_raw['inv2_kwh'] + bissell_raw['inv3_kwh']
)
mismatch = (abs(bissell_raw['inv_sum'] - bissell_raw['total_kwh']) > 0.5).sum()
print(f"   Inverter sum vs total mismatch: {mismatch} rows "
      f"({'✅ data is consistent' if mismatch == 0 else '⚠ check data'})")

zero_rows_b = (bissell_raw['total_kwh'] == 0).sum()
print(f"   Zero-production days: {zero_rows_b} rows removed "
      f"(no generation — nighttime/offline)")

bissell_long = bissell_raw.melt(
    id_vars=['date', 'total_kwh'],
    value_vars=['inv1_kwh', 'inv2_kwh', 'inv3_kwh'],
    var_name='inv_raw',
    value_name='daily_yield_kwh'
)
bissell_long['unit_id'] = bissell_long['inv_raw'].map({
    'inv1_kwh': 'Bissell-INV-1',
    'inv2_kwh': 'Bissell-INV-2',
    'inv3_kwh': 'Bissell-INV-3'
})
bissell_long['source']          = 'Bissell'
bissell_long['installed_kw']    = 7.6
bissell_long['full_load_hours'] = np.nan
bissell_long['monthly_yield']   = np.nan
bissell_long['plant_status']    = 'Active'
bissell_long['earning_cad']     = np.nan

bissell_long = bissell_long[bissell_long['daily_yield_kwh'] > 0].copy()

bissell_df = bissell_long[[
    'date', 'unit_id', 'source', 'daily_yield_kwh',
    'installed_kw', 'full_load_hours', 'monthly_yield',
    'plant_status', 'earning_cad'
]].copy()

print(f"\n   ✅ Bissell clean: {len(bissell_df):>4} daily records "
      f"| Units : {bissell_df['unit_id'].unique().tolist()}")

combined = pd.concat([visser_df, bissell_df], ignore_index=True)
combined  = combined.sort_values(['unit_id', 'date']).reset_index(drop=True)

combined_dups = combined.duplicated(subset=['date','unit_id']).sum()
if combined_dups:
    print(f"   ⚠ {combined_dups} duplicate unit-date rows in combined set → removed")
    combined = combined.drop_duplicates(subset=['date','unit_id'])

print(f"""
┌─────────────────────────────────────────────────────┐
│              DATA QUALITY REPORT                    │
├──────────────────────────┬──────────────────────────┤
│  Visser clean records    │  {len(visser_df):>6}                  │
│  Bissell clean records   │  {len(bissell_df):>6}                  │
│  Combined total          │  {len(combined):>6}                  │
│  Units monitored         │  {combined['unit_id'].nunique():>6}                  │
│  Date range              │  {str(combined['date'].min().date())} → {str(combined['date'].max().date())}  │
│  Missing yield values    │  {combined['daily_yield_kwh'].isna().sum():>6}                  │
│  Negative yield values   │  {(combined['daily_yield_kwh'] < 0).sum():>6}                  │
│  Remaining duplicates    │  {combined.duplicated(subset=['date','unit_id']).sum():>6}                  │
└──────────────────────────┴──────────────────────────┘
""")

print("="*65)
print("  PHASE 2: FEATURE ENGINEERING")
print("="*65)

df = combined.copy()

df['rolling_mean_7'] = (
    df.groupby('unit_id')['daily_yield_kwh']
      .transform(lambda x: x.rolling(7, min_periods=2).mean())
)
df['rolling_std_7'] = (
    df.groupby('unit_id')['daily_yield_kwh']
      .transform(lambda x: x.rolling(7, min_periods=2).std().fillna(0))
)

df['z_score'] = (
    df.groupby('unit_id')['daily_yield_kwh']
      .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
)

fleet_avg = (
    df.groupby('date', as_index=False)
      .agg(fleet_avg_yield=('daily_yield_kwh', 'mean'))
)
df = df.merge(fleet_avg, on='date', how='left')

df['pct_dev_from_fleet'] = (
    (df['daily_yield_kwh'] - df['fleet_avg_yield'])
    / (df['fleet_avg_yield'] + 1e-9)
) * 100

df['day_change_pct'] = (
    df.groupby('unit_id')['daily_yield_kwh']
      .pct_change().fillna(0) * 100
)

df['lag_1'] = df.groupby('unit_id')['daily_yield_kwh'].shift(1)
df['lag_7'] = df.groupby('unit_id')['daily_yield_kwh'].shift(7)

df['ratio_to_rollmean'] = (
    df['daily_yield_kwh'] / (df['rolling_mean_7'] + 1e-9)
)

df['performance_ratio'] = (
    df['daily_yield_kwh'] / (df['installed_kw'] + 1e-9)
)

feature_df = df.dropna(subset=['rolling_mean_7', 'lag_1']).copy().reset_index(drop=True)
print(f"✅ Features created | {len(feature_df)} rows | {feature_df['unit_id'].nunique()} units")

print("\n" + "="*65)
print("  PHASE 3: ANOMALY DETECTION")
print("="*65)

feature_df['rule_anomaly'] = (
    (feature_df['z_score'] < -1.8) |
    (feature_df['pct_dev_from_fleet'] < -20) |
    (feature_df['day_change_pct'] < -30) |
    (feature_df['ratio_to_rollmean'] < 0.65)
).astype(int)
print(f"✅ Rule-based anomalies   : {feature_df['rule_anomaly'].sum()}")

model_features = [
    'daily_yield_kwh', 'rolling_mean_7', 'rolling_std_7',
    'z_score', 'pct_dev_from_fleet', 'day_change_pct',
    'ratio_to_rollmean', 'performance_ratio'
]
X_raw    = feature_df[model_features].fillna(0)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

iso = IsolationForest(contamination=0.07, random_state=42, n_estimators=200)
feature_df['iso_pred']   = iso.fit_predict(X_scaled)
feature_df['ml_anomaly'] = (feature_df['iso_pred'] == -1).astype(int)
print(f"✅ ML (Isolation Forest)  : {feature_df['ml_anomaly'].sum()}")

feature_df['anomaly_flag'] = (
    (feature_df['rule_anomaly'] == 1) | (feature_df['ml_anomaly'] == 1)
).astype(int)
total_anoms = int(feature_df['anomaly_flag'].sum())
print(f"✅ Combined anomaly flags : {total_anoms} "
      f"({total_anoms / len(feature_df) * 100:.1f}% of records)")

rf = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight='balanced'
)
rf.fit(X_scaled, feature_df['rule_anomaly'])
feat_imp = pd.DataFrame({
    'feature':    model_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(feat_imp['feature'], feat_imp['importance'],
               color='#5C6BC0', edgecolor='white')
for bar, val in zip(bars, feat_imp['importance']):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_title('Feature Importance — Random Forest Classifier',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig('images/chart_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
feat_imp.to_csv('feature_importance.csv', index=False)
print("✅ Feature importance chart + CSV saved")

print("\n" + "="*65)
print("  PHASE 4: EXPLAINABLE AI")
print("="*65)

def explain_row(row):
    if row['anomaly_flag'] == 0:
        return 'Normal Operation'
    reasons = []
    if row['z_score'] < -1.8:
        reasons.append(f"Low vs own history (z={row['z_score']:.2f})")
    if row['pct_dev_from_fleet'] < -20:
        reasons.append(
            f"{abs(row['pct_dev_from_fleet']):.0f}% below fleet avg")
    if row['day_change_pct'] < -30:
        reasons.append(
            f"Sharp drop from yesterday ({row['day_change_pct']:.0f}%)")
    if row['ratio_to_rollmean'] < 0.65:
        reasons.append(
            f"Only {row['ratio_to_rollmean']*100:.0f}% of 7-day avg")
    if row['ml_anomaly'] == 1 and not reasons:
        reasons.append('ML flagged unusual multi-metric pattern')
    return ' | '.join(reasons)

feature_df['explanation'] = feature_df.apply(explain_row, axis=1)

print("\n📋 Anomaly count by unit:")
print(feature_df.groupby(['source', 'unit_id'])['anomaly_flag']
      .sum().sort_values(ascending=False).to_string())

feature_df.to_csv('final_anomaly_results.csv', index=False)
print(f"\n✅ Saved: final_anomaly_results.csv "
      f"({len(feature_df)} rows, {len(feature_df.columns)} columns)")

print(f"""
╔══════════════════════════════════════════════════════════╗
║               PIPELINE COMPLETE — SUMMARY               ║
╠══════════════════════════════════════════════════════════╣
║  Records processed   : {len(feature_df):>6}                          ║
║  Anomalies detected  : {total_anoms:>6}                          ║
║  Rule-based flags    : {int(feature_df['rule_anomaly'].sum()):>6}                          ║
║  ML flags            : {int(feature_df['ml_anomaly'].sum()):>6}                          ║
║  Units monitored     : {feature_df['unit_id'].nunique():>6}                          ║
║  Date range          : {str(feature_df['date'].min().date())} → {str(feature_df['date'].max().date())}    ║
╚══════════════════════════════════════════════════════════╝
""")