# In questo script, carichiamo test_short_clean che contiene i report reali, e carichiamo generated_report, che contiene i report generati dal modello.
# In ogni colonna di generated_report, ci sono i report generati da un checkpoint diverso.
# calcoliamo il BLEU score per ogni checkpoint

import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sacrebleu.metrics import BLEU

"""
# Carichiamo i report generati
a = pd.read_csv('generated_report_0.csv')
b = pd.read_csv('generated_report_10.csv')
c = pd.read_csv('generated_report_20.csv')
d = pd.read_csv('generated_report_30.csv')
e = pd.read_csv('generated_report_40.csv')
f = pd.read_csv('generated_report_50.csv')
g = pd.read_csv('generated_report_60.csv')
h = pd.read_csv('generated_report_70.csv')
i = pd.read_csv('generated_report_80.csv')
j = pd.read_csv('generated_report_90.csv')
k = pd.read_csv('generated_report_99.csv')

# Creiamo un dataframe vuoto con colonne Real, 0, 10, 20, 30, 40, 50, 60, 70
generated_report = pd.DataFrame(columns=['Real', '0', '10', '20', '30', '40', '50', '60', '70'])
generated_report['Real'] = a['Real']
generated_report['0'] = a['0']
generated_report['10'] = b['numbers']
generated_report['20'] = c['20']
generated_report['30'] = d['numbers']
generated_report['40'] = e['numbers']
generated_report['50'] = f['numbers']
generated_report['60'] = g['60']
generated_report['70'] = h['numbers']
generated_report['80'] = i['numbers']
generated_report['90'] = j['numbers']
generated_report['99'] = k['numbers']

generated_report.to_csv('generated_report.csv', index=False)
"""
views = pd.read_csv('test_short_clean.csv')['ViewPosition']
generated_report = pd.read_csv('generated_report.csv')
generated_report['ViewPosition'] = views

view_to_test = 'Lateral'
if view_to_test == 'Frontal':
    generated_report = generated_report[generated_report['ViewPosition'].isin(['AP', 'PA'])]
elif view_to_test == 'Lateral':
    generated_report = generated_report[generated_report['ViewPosition'].isin(['LATERAL', 'LL'])]

hypothesis = generated_report['Real']
hypothesis = [h.lower() for h in hypothesis]
hypothesis_tok = [[nltk.word_tokenize(h)] for h in hypothesis]

# Sostituiamo i NaN con stringhe vuote
generated_report = generated_report.fillna('')
# Proviamo a calcolarlo come sentence_bleu
to_try = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
bleu_corpus_1 = {x: 0 for x in to_try}
bleu_corpus_2 = {x: 0 for x in to_try}
bleu_corpus_3 = {x: 0 for x in to_try}
bleu_corpus_4 = {x: 0 for x in to_try}

smoothing_function = SmoothingFunction().method1

for i in to_try:
    generated = generated_report[str(i)]
    generated = [g.lower() for g in generated]
    generated_tok = [nltk.word_tokenize(g) for g in generated]
    bleu_score_1 = corpus_bleu(hypothesis_tok, generated_tok, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
    bleu_score_2 = corpus_bleu(hypothesis_tok, generated_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
    bleu_score_3 = corpus_bleu(hypothesis_tok, generated_tok, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
    bleu_score_4 = corpus_bleu(hypothesis_tok, generated_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
    bleu_corpus_1[int(i)] = bleu_score_1
    bleu_corpus_2[int(i)] = bleu_score_2
    bleu_corpus_3[int(i)] = bleu_score_3
    bleu_corpus_4[int(i)] = bleu_score_4

"""
# Calcoliamolo anche con SacreBLEU
to_try = [0, 10, 20, 30, 40, 50, 60, 70, 80, 99]
bleu_sacre = {x: 0 for x in to_try}
bleu = BLEU()

refs = [hypothesis]

for i in to_try:
    generated = generated_report[str(i)]
    generated = [g.lower() for g in generated]
    bleu_score = bleu.corpus_score(generated, refs)
    bleu_sacre[int(i)] = bleu_score.score
"""