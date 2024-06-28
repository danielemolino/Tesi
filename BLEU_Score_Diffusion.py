# In questo codice, dati i vari checkpoint del VAE, calcoliamo per ciascuno il BLEU score
# e salviamo il risultato in un file .csv

import torch
import torch.nn.functional as F
import nltk
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from core.models.model_module_infer import model_module
from core.models.dani_model import dani_model
from Clip_Training.DataLoader import MIMIC_CXR_Dataset
from omegaconf import OmegaConf
import os
import tifffile
import re
from tqdm import tqdm


def extract_numbers(string):
    return re.findall(r'\d+', string)


# Funzione per caricare i checkpoint e valutare i testi
def generate(inference_tester, dataloader, numbers, device):
    inference_tester.eval()
    # creiamo un df per salvare i risultati
    df = pd.DataFrame(columns=['Real', 'numbers'])
    for batch in tqdm(dataloader):
        with torch.no_grad():
            images, texts, subject_ids, study_ids, paths = batch
            images = images.unsqueeze(1).float()
            texts = list(texts)
            paths = list(paths)
            n_samples = len(paths)
            # Ora invece usiamo im_real come conditioning per generare il testo
            conditioning = []
            cim = inference_tester.net.clip(images.to(device), encode_type='encode_vision')
            uim = None
            scale = 7.5
            dummy = torch.zeros_like(im_real).to(device)
            uim = inference_tester.net.clip_encode_vision(im_real, encode_type='encode_vision').to(device)
            conditioning.append(torch.cat([uim, cim]))

            shapes = []
            n = 768
            shape = [1, n]
            shapes.append(shape)

            z, _ = inference_tester.sampler.sample(
                steps=50,
                shape=shapes,
                condition=conditioning,
                unconditional_guidance_scale=scale,
                xtype=['text'],
                condition_types=['frontal'],
                eta=1,
                verbose=False,
                mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

            # adesso la passiamo al decoder
            x = inference_tester.net.optimus_decode(z[0], max_length=100)
            x = [a.tolist() for a in x]
            rec_text = [inference_tester.net.optimus.tokenizer_decoder.decode(a) for a in x]
            # elimina i token speciali che sono <BOS> e <EOS>
            rec_text = rec_text[0].replace('<BOS>', '').replace('<EOS>', '')
            print('REAL: ' + text)
            print('GEN: ' + rec_text)
            df = df._append({'Real': text, 'numbers': rec_text}, ignore_index=True)
    df.to_csv(f'csv/generated_report_{numbers[0]}.csv', index=False)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #unet_weights = os.listdir(f'Report_Training/saved_checkpoints/Unet')
    #unet_weights = [f'Report_Training/saved_checkpoints/Unet/{x}' for x in unet_weights if x.endswith('.pt')]

    model_load_paths = ['CoDi_encoders.pth']
    inference_tester = dani_model(model='thesis_model', data_dir='checkpoints/', pth=model_load_paths,
                                  load_weights=False)  # turn on fp16=True if loading fp16 weights

    clip_weights = 'Clip_Training/saved_checkpoints/Training_Clip_5e^-5/checkpoint_29_epoch_Training_Clip_5e^-5.pt'
    #a, b = inference_tester.net.clip.load_state_dict(torch.load(clip_weights, map_location=device), strict=False)

    optimus_weights = 'Report_Training/saved_checkpoints/VAE/checkpoint_99_epoch_VAE-Training-Prova1.pt'
    #optimus_weights = torch.load(optimus_weights, map_location='cpu')
    #a, b = inference_tester.net.optimus.load_state_dict(optimus_weights, strict=False)

    # Load the dataloader
    path_to_csv = 'csv/test_short_clean.csv'
    csv = pd.read_csv(path_to_csv)
    dataset = MIMIC_CXR_Dataset(csv, '256/')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    unet_weights = [unet_weights[0]]
    for w in unet_weights:
        numbers = extract_numbers(w)
        print(numbers)
        w1 = torch.load(w, map_location='cpu')
        a, b = inference_tester.net.model.load_state_dict(w1, strict=False)
        # evaluate
        generate(inference_tester, dataloader, numbers, device)


if __name__ == '__main__':
    main()

