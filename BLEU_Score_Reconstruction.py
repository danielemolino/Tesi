# In questo codice, dati i vari checkpoint del VAE, calcoliamo per ciascuno il BLEU score
# e salviamo il risultato in un file .csv

import torch
import torch.nn.functional as F
import nltk
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from core.models.model_module_infer import model_module
from omegaconf import OmegaConf
import os

nltk.download('punkt')


# Funzione per calcolare il BLEU score per un batch di testi
def calculate_bleu_score(original_texts, reconstructed_texts):
    bleu_scores = []
    for original, reconstructed in zip(original_texts, reconstructed_texts):
        # facciamo lowercase ai testi
        original = original.lower()
        reconstructed = reconstructed.lower()
        reference = [nltk.word_tokenize(original)]
        candidate = nltk.word_tokenize(reconstructed)
        bleu_score = sentence_bleu(reference, candidate)
        bleu_scores.append(bleu_score)
    return sum(bleu_scores) / len(bleu_scores)


# Funzione per caricare i checkpoint e valutare i testi
def BLEU(model, test_set):
    model.eval()

    # Ricostruisci i testi dal test set
    original_texts = []
    reconstructed_texts = []
    test_set = test_set[:3]
    with torch.no_grad():
        for text in test_set:
            c = model.optimus_encode([text])
            d = model.optimus_decode(c, max_length=77)
            d = [x.tolist() for x in d]
            rec_text = [model.optimus.tokenizer_decoder.decode(x) for x in d]
            # elimina i token speciali che sono <BOS> e <EOS>
            rec_text = rec_text[0].replace('<BOS>', '').replace('<EOS>', '')
            original_texts.append(text)
            reconstructed_texts.append(rec_text)
    # Calcola il BLEU score
    bleu_score = calculate_bleu_score(original_texts, reconstructed_texts)
    return bleu_score


def main():
    optimus_weights = os.listdir(f'Report_Training/saved_checkpoints/VAE')
    optimus_weights = [f'CXR_Training/saved_checkpoints/VAE/{x}' for x in optimus_weights if x.endswith('.pt')]
    # eliminiamo tutti i pesi che sono già presenti nel csv CXR_Lateral_FID_Xray.csv
    if os.path.exists(f'csv/VAE_BLEU_Reconstruction.csv'):
        df = pd.read_csv(f'csv/VAE_BLEU_Reconstruction.csv')
        weights = df['Weight'].tolist()
        optimus_weights = [x for x in optimus_weights if x not in weights]

    model_load_paths = ['CoDi_encoders.pth']
    inference_tester = model_module(model='codi', load_weights=True, data_dir='checkpoints/', pth=model_load_paths,
                                    fp16=False)
    codi = inference_tester.net
    codi.audioldm = None
    codi.clap = None
    del inference_tester

    # Load the dataloader
    path_to_csv = 'csv/test_short_clean.csv'
    csv = pd.read_csv(path_to_csv)
    # campioniamo 500 testi con seed 42
    csv = csv.sample(500, random_state=42)['report'].tolist()

    for w in optimus_weights:
        w = torch.load(w, map_location='cpu')
        a, b = codi.optimus.load_state_dict(w, strict=False)

        # evaluate
        bleu = BLEU(codi, csv)

        output_path = f'csv/VAE_BLEU_Reconstruction.csv'
        # Se non esiste, creiamolo, deve avere le colonne Weight e  FID
        if not os.path.exists(output_path):
            df = pd.DataFrame(columns=['Weight', 'BLEU'])
            df.to_csv(output_path, index=False)
        df = pd.read_csv(output_path)
        df = df._append({'Weight': w, 'BLEU': bleu}, ignore_index=True)
        df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()

