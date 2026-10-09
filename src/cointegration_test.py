import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from data_loader import fetch_multiple_stocks_close
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

'''
def test_cointegration(series_y, series_x, sig_level=0.05):
    
    Tests if two price series are cointegrated using the Engle-Granger two step test (based on ADF test)
    

    score, p_value, crit_values = coint(series_y, series_x)

    is_cointegrated = p_value < sig_level

    return{
    'p_value': p_value,
    't_stat': score,
    'critical_values': crit_values,
    'is_cointegrated': is_cointegrated, 
    }

'''

df = fetch_multiple_stocks_close(["KO", "PEP"], "2020-01-01", "2024-01-01")

#COINTEGRATION TEST
logp = np.log(df[['KO', 'PEP']].dropna())
split = "2021-12-31"
form = logp.loc[:split]

_, pval, _ = coint(form['KO'], form['PEP'])
ols = sm.OLS(form['KO'], sm.add_constant(form['PEP'])).fit()
a, beta = ols.params['const'], ols.params['PEP']

spread = logp['KO'] - (a + beta *logp['PEP'])
z = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
z_trade = z.loc[split:].iloc[1:]

#HALF LIFE
s = spread.loc[:split]
d = pd.DataFrame({"ds": s.diff(), "lag": s.shift(1)}).dropna()

fit = sm.OLS(d["ds"], sm.add_constant(d["lag"])).fit()
b, se = fit.params["lag"], fit.bse["lag"]

def half_life(b):
    return -np.log(2) / np.log(1 + b) if -1 < b < 0 else np.inf

print(f"b = {b:.4f}, t-stat = {b/se:.2f}")
print(f"half-life: {half_life(b):.1f} days")
print(f"rough range: {half_life(b - 2*se):.1f} to {half_life(b + 2*se):.1f} days")

q = input("Show plots y/n").strip().lower()
if q == 'y':
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(logp.index, spread, label='Spread plot')

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True, which='major', axis='x', alpha=0.4, linestyle='-')
    ax.grid(True, which='minor', axis='x', alpha=0.15, linestyle=':')

    ax.axhline(0, c='k', ls='--')
    ax.set_title("Spread plot")
    ax.set_ylabel("Spread")
    ax.set_xlabel("Date")
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

