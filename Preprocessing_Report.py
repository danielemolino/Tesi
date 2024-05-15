# Per quanto riguarda il preprocessing dei report, la logica da seguire è la seguente:
# 1. Caricare i report
# 2. Estrarre da ogni report la sezione "findings" e "impression" in due variabili
# 3. Salvare le due come due txt separati

import os
import pandas as pd
from tqdm import tqdm

root_dir = 'E:/CXR/physionet.org/files/mimic-cxr/2.0.0/files/'
dest_dir_256 = 'E:/CXR_256/'
dest_dir_512 = 'E:/CXR_512/'
csv = pd.read_csv('Mimic-CXR.csv')

csv['subject_id'] = csv['subject_id'].astype(str)
csv = csv[csv['subject_id'].str.startswith('10')]


for i in tqdm(range(len(csv))):
    row = csv.iloc[i]
    report_path = os.path.join(root_dir, row['path'])
    report_path = report_path[:report_path.rfind('/')] + '.txt'
    if not os.path.exists(report_path):
        continue
    path = 'p' + row['subject_id'][:2] + '/p' + row['subject_id']+ '/s' + row['study_id'].astype(str)
    dest_path_256 = os.path.join(dest_dir_256, path)
    dest_path_512 = os.path.join(dest_dir_512, path)
    # Se i file esistono già, continua
    if os.path.exists(dest_path_256 + '.tiff') and os.path.exists(dest_path_512 + '.tiff'):
        continue
    # Carichiamo il report
    with open(report_path, 'r') as f:
        report = f.read()
    # Estraiamo le sezioni
    findings = report[report.find('FINDINGS:') + 9:report.find('IMPRESSION:')].strip()
    impression = report[report.find('IMPRESSION:') + 11:].strip()

    if not os.path.exists(os.path.dirname(dest_path_256)):
        os.makedirs(os.path.dirname(dest_path_256))
    if not os.path.exists(os.path.dirname(dest_path_512)):
        os.makedirs(os.path.dirname(dest_path_512))
    with open(dest_path_256 + '_findings.txt', 'w') as f:
        f.write(findings)
    with open(dest_path_256 + '_impression.txt', 'w') as f:
        f.write(impression)
    with open(dest_path_512 + '_findings.txt', 'w') as f:
        f.write(findings)
    with open(dest_path_512 + '_impression.txt', 'w') as f:
        f.write(impression)