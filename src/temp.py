import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from data_loader import fetch_stock_data

ko_data = fetch_stock_data("KO", "2020-01-01", "2024-01-01")
pep_data = fetch_stock_data("PEP", "2020-01-01", "2024-01-01")

df = pd.DataFrame({
    'KO': ko_data['Close'].squeeze(),
    'PEP': pep_data['Close'].squeeze()
}).dropna()

print(df['KO'].head())