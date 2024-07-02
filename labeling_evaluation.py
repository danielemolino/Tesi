# Questo script, prende un csv di report, e li inserisce in un csv headerless in cui tutti i report che contengono:
# - una virgola
# - si estendono su più righe
# Vengono racchiuse tra virgolette.

import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, classification_report, fbeta_score
from tqdm import tqdm
import os

to_elaborate = ['Real', 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
labeler = 'chexbert'
columns = ['weights', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
other_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
# Creiamo un csv per l'F1 score
f1_score_df = pd.DataFrame(columns=columns)

for w in to_elaborate:
    # Creiamo un dizionario con chiavi columns e valori 0
    f1_score_dict = {k: 0 for k in columns}
    f1_score_dict['weights'] = w
    # Carichiamo il csv
    df = pd.read_csv(f'{labeler}_results/scale_{labeler}_labeled_reports_{w}.csv')
    # In tutte le righe in cui no finding è 1, mettiamo tutti gli altri a 0 tranne no finding
    df.loc[df['No Finding'] == 1, other_columns] = 0

    labels = pd.read_csv('labels_full_test.csv')
    # rendiamo i -1 come NaN in entrabi i dataframe
    df = df.replace(-1, None)
    labels = labels.replace(-1, None)

    # Ora calcoliamo per ogni colonna l'F1 score
    for disease in columns[1:]:
        y = labels[disease].rename(f'{disease}_y')
        x = df[disease].rename(f'{disease}_x')

        conc = pd.concat([y, x], axis=1)
        conc = conc.dropna(subset=[f'{disease}_y'])
        conc = conc.dropna(subset=[f'{disease}_x'])

        """
        # Prendiamo tutti gli indici in cui y è 1
        idx = conc[conc[f'{disease}_y'] == 1].index
        idx_0 = conc[conc[f'{disease}_y'] == 0].index
        # Peschiamone 4000 da idx_0, solo che è un index object, sample non funziona
        idx_0 = idx_0.to_list()
        idx_0 = idx_0[:1000]
        idx_0 = pd.Index(idx_0)
        idx = idx.union(idx_0)
        conc = conc.loc[idx]
        """

        # calcoliamo pos e tot che sono il numero di positivi di y e il numero di sample di x
        pos = conc[f'{disease}_y'].sum()
        tot = conc[f'{disease}_y'].count()
        #print(f'Pathology: {disease}, Positives: {pos}, Total: {tot}')

        # Rendiamo tutto Int
        conc[f'{disease}_y'] = conc[f'{disease}_y'].astype(int)
        conc[f'{disease}_x'] = conc[f'{disease}_x'].astype(int)

        # Calcoliamo l'F1 score
        f1 = f1_score(conc[f'{disease}_y'], conc[f'{disease}_x'])
        f1_score_dict[disease] = f1

        # Calcoliamo il classification report
        print(f'{disease} Classification Report:')
        print(classification_report(conc[f'{disease}_y'], conc[f'{disease}_x']))

    f1_score_df = f1_score_df._append(f1_score_dict, ignore_index=True)
f1_score_df.to_csv(f'{labeler}_results/{labeler}_f1_score.csv', index=False)