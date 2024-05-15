# In questo script dobbiamo dividere il dataset in train e test set
import pandas as pd
import random

# Load the dataset
csv = pd.read_csv('Mimic-CXR.csv', dtype={'subject_id': str, 'study_id': str, 'ViewPosition': str})
csv = csv.dropna(subset=['ViewPosition'])
csv = csv[csv['ViewPosition'].isin(['AP', 'PA', 'LATERAL', 'LL'])]

# Le colonne che mi interessano principalmente sono 'subject_id' e 'study_id'
# Infatti, devo evitare che un paziente sia presente con uno studio diverso sia nel train che nel test set

csv_multiindex = csv.set_index(['subject_id', 'study_id'])
csv_subjects = csv.set_index(['subject_id'])
csv_studies = csv.set_index(['study_id'])

# Vogliamo garantire che un paziente sia presente solo nel train set o nel test set, e che tutti quelli del test abbiano una vista frontale e una laterale
# Facciamo dapprima un check, filtriamo tutti gli studi che non hanno una vista frontale e una laterale
study_ids = csv['study_id'].unique()
map = csv.groupby(['study_id']).apply(lambda x: (x['ViewPosition'].isin(['AP', 'PA'])).any() and (x['ViewPosition'].isin(['LATERAL', 'LL']).any()))
map = map[map].index.tolist()
map = csv['study_id'].isin(map)
csv_filtered = csv[map]

subject_ids = csv['subject_id'].unique()
subject_ids_filtered = csv_filtered['subject_id'].unique()

test_portion = int(round(0.2*len(subject_ids)))

# Creiamo il test set
seed = 11
random.seed(seed)
test_subject_ids = random.sample(list(subject_ids_filtered), test_portion)
test_csv = csv[csv['subject_id'].isin(test_subject_ids)]
