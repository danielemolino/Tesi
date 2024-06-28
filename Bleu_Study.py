# Carichiamo il csv generated_report.csv

import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sacrebleu.metrics import BLEU
from tqdm import tqdm

df = pd.read_csv('generated_report.csv')
df = df.fillna('')

views = pd.read_csv('test_short_clean.csv')['ViewPosition']
df['ViewPosition'] = views

view_to_test = 'Lateral'
if view_to_test == 'Frontal':
    df = df[df['ViewPosition'].isin(['AP', 'PA'])]
elif view_to_test == 'Lateral':
    df = df[df['ViewPosition'].isin(['LATERAL', 'LL'])]

# Adesso, per ogni studio, calcoliamo il BLEU score tra tutti i report di quello studio, per capire se mediamente i report di uno stesso studio sono simili tra loro
studies = df['Study_id'].unique()

to_try = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
bleu_corpus_1 = {x: 0 for x in to_try}
bleu_corpus_2 = {x: 0 for x in to_try}
bleu_corpus_3 = {x: 0 for x in to_try}
bleu_corpus_4 = {x: 0 for x in to_try}

smoothing_function = SmoothingFunction().method1
for i in tqdm(to_try):
    n_studies = 0
    for study in tqdm(studies):
        generated = df[df['Study_id'] == study][str(i)]
        if len(generated) < 2:
            continue
        n_studies += 1
        generated = [g.lower() for g in generated]
        generated_tok = [nltk.word_tokenize(g) for g in generated]

        # Adesso, non ho una ground truth, perchè il mio obiettivo è valutare la similarità intra-studio, quindi prendo tutti i report dello studio e li metto in una lista
        # Allora, provo tutte le combinazioni di reference vs hypothesis, quindi ogni report verrà confrontato con tutti gli altri
        # Se ho ad esempio 4 report, avrò il test 1 vs 2-3-4 poi 2 vs 1-3-4 poi 3 vs 1-2-4 poi 4 vs 1-2-3
        reports = [i for i in range(len(generated))]
        bleu_score_1 = 0
        bleu_score_2 = 0
        bleu_score_3 = 0
        bleu_score_4 = 0
        for report in reports:
            reference = [generated_tok[report]]
            hypothesis = [generated_tok[:report] + generated_tok[report+1:]]
            bleu_score_1 += corpus_bleu(hypothesis, reference, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
            bleu_score_2 += corpus_bleu(hypothesis, reference, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
            bleu_score_3 += corpus_bleu(hypothesis, reference, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
            bleu_score_4 += corpus_bleu(hypothesis, reference, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

        bleu_corpus_1[int(i)] += bleu_score_1/len(reports)
        bleu_corpus_2[int(i)] += bleu_score_2/len(reports)
        bleu_corpus_3[int(i)] += bleu_score_3/len(reports)
        bleu_corpus_4[int(i)] += bleu_score_4/len(reports)

    bleu_corpus_1[int(i)] /= n_studies
    bleu_corpus_2[int(i)] /= n_studies
    bleu_corpus_3[int(i)] /= n_studies
    bleu_corpus_4[int(i)] /= n_studies

for i in to_try:
    print(f'BLEU-1: {i} -> {bleu_corpus_1[i]}')
    print(f'BLEU-2: {i} -> {bleu_corpus_2[i]}')
    print(f'BLEU-3: {i} -> {bleu_corpus_3[i]}')
    print(f'BLEU-4: {i} -> {bleu_corpus_4[i]}')
    print('')