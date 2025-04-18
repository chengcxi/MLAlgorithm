import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pl.read_csv('fundamentals_data.csv')
df = df.with_columns(pl.col('Date').str.to_datetime())
df_pd = df.to_pandas().set_index('Date')
df_pd = df_pd.sort_values(by='Date')
if df_pd.isnull().values.any():
    print("STILL NULL")
    df_pd = df_pd.bfill().ffill().fillna(0)
scaler = StandardScaler()
noscale = ['Date']
doscale = [col for col in df_pd.columns if col not in noscale]
scaled_data = scaler.fit_transform(df_pd[doscale])

pca = PCA(n_components=.9,)
pca_data = pca.fit_transform(scaled_data)
print(pca_data.shape)

pca_results = pd.DataFrame(
    pca_data,
    index=df_pd.index,
    columns=[f'PC_{i+1}' for i in range(pca_data.shape[1])]
)
print(pca_results)
pca_results.to_csv('pca_results.csv')