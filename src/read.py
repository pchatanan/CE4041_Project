import pandas as pd
import numpy as np

file_names = ['XGBoost200.csv', 'XGBoost350.csv', 'XGBoost500.csv', 'XGBoost650.csv']

dfs = []
for name in file_names:
    dfs.append(pd.read_csv(name))

all_df = pd.concat(dfs, axis=1)

print(all_df.head(5))

all_df['loss'] = np.mean(all_df.iloc[:, [1, 3, 5, 7]], axis=1)

print(all_df.head(5))

all_df.iloc[:, :2].to_csv('yeah.csv', index=False)
