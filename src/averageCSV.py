import pandas as pd
import numpy as np
import glob

file_names = glob.glob("*.csv")
cols = np.arange(1,2*len(file_names)-1,2)

dfs = []
for name in file_names:
    dfs.append(pd.read_csv(name))

all_df = pd.concat(dfs, axis=1)
all_df['loss'] = np.mean(all_df.iloc[:, cols], axis=1)
all_df.iloc[:, :2].to_csv('submission.csv', index=False)
