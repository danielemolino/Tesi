import os
import pandas as pd
from tqdm import tqdm

df_24h_literature = pd.read_csv('Mimic-CXR.csv')
df_24h_literature['subject_id'] = df_24h_literature['subject_id'].astype(str)
df_24h_literature = df_24h_literature[df_24h_literature['subject_id'].str.startswith('10')]

# per ogni riga di df_24h_literature, controlliamo se i file sono presenti sull'hard disk, quelli che non ci sono li aggiungimao a un altro dataframe
# creiamo un dataframe vuoto
for i in tqdm(range(len(df_24h_literature))):
    path1 = '256/' + df_24h_literature['path'][i][:-4] + '.tiff'
    path2 = '512/' + df_24h_literature['path'][i][:-4] + '.tiff'

    if not os.path.exists(path1):
        print(path1)
    if not os.path.exists(path2):
        print(path2)

a = pd.read_csv('missing_files256.csv')
b = pd.read_csv('missing_files512.csv')
