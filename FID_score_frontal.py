# In questo codice, analizzeremo le performance della U-Net di CoDi addestrata sul nostro subset di dati di xrays

# Iniziamo caricando il modello
import os
import pandas as pd
from core.models.model_module_infer import model_module
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from Clip_Training.DataLoader import MIMIC_CXR_Dataset


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(torch.cuda.device_count())

model_load_paths = ['CoDi_encoders.pth', 'CoDi_text_diffuser.pth', 'CoDi_video_diffuser_8frames.pth']
inference_tester = model_module(data_dir='checkpoints/', pth=model_load_paths,
                                fp16=False, load_weights=True)  # turn on fp16=True if loading fp16 weights
inference_tester = inference_tester.to(device)
inference_tester.eval()
# prendiamo tutti i .pt nella cartella saved_checkpoints con os.listddir
unet_weights = os.listdir('CXR_Training/saved_checkpoints')
unet_weights = [f'CXR_Training/saved_checkpoints/{x}' for x in unet_weights if x.endswith('.pt')]
clip_weights = 'Clip_Training/saved_checkpoints/DirtyData-20Ep-5e^-5/checkpoint_19_epoch_DirtyData-20Ep-5e^-5.pt'
a, b = inference_tester.net.clip.load_state_dict(torch.load(clip_weights, map_location=device), strict=False)

csv = pd.read_csv('test_short_frontal.csv')
dataset = MIMIC_CXR_Dataset(csv, '256/')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
labels = pd.read_csv('labels.csv')

n_test = 5
correct_matches = 0
correct_class_matches = 0
j = 0
unet_weights = unet_weights[0]
a, b = inference_tester.net.model.load_state_dict(torch.load(unet_weights, map_location=device), strict=False)
print(f'++++++ {unet_weights} ++++++')
for batch in dataloader:
    j += 1
    _, true_text, subject_id, study_id = batch
    subject_id = int(subject_id.numpy())
    study_id = int(study_id.numpy())

    mask = csv['subject_id'] != subject_id
    csv2 = csv[mask]

    csv2 = csv2.drop_duplicates(subset='study_id')
    dataset2 = MIMIC_CXR_Dataset(csv2, '256/')
    dataloader2 = DataLoader(dataset2, batch_size=10, shuffle=True)

    _, texts, subject_ids, study_ids = next(iter(dataloader2))
    study_ids = study_ids.numpy()
    texts = true_text + texts
    # Per ciascuno dei report, dobbiamo passarlo al modello per generare un campione sintetico
    images = []
    for text in texts:
        with torch.no_grad():
            ctx = inference_tester.net.clip_encode_text(1 * [text], encode_type='encode_text').to(device)
            utx = None
            scale = 2.0
            conditioning = []
            if scale != 1.0:
                utx = inference_tester.net.clip_encode_text(1 * [""], encode_type='encode_text').to(device)
            conditioning.append(torch.cat([utx, ctx]))

            h, w = [256, 256]
            shapes = []
            shape = [1, 4, h // 8, w // 8]
            shapes.append(shape)

            z, _ = inference_tester.sampler.sample(
                steps=2,
                shape=shapes,
                condition=conditioning,
                unconditional_guidance_scale=scale,
                xtype=['image'],
                condition_types=['text'],
                eta=1,
                verbose=False,
                mix_weight={'video': 1, 'audio': 1, 'text': 1, 'image': 1})

            image = inference_tester.net.autokl_decode(z[0])
            x = torch.clamp((image[0] + 1.0) / 2.0, min=0.0, max=1.0)
            images.append(x)
    images = torch.stack(images)
    # uniamo study_id e study_ids
    study_ids = np.concatenate([[study_id], study_ids])

    with torch.no_grad():
        texts_features = inference_tester.net.clip_encode_text(list(texts), encode_type='encode_text')

        images_features = inference_tester.net.clip_encode_vision(images, 'encode_vision')
        # calcoliamo la similarità tra la query e tutti i report
        # normalizziamo i vettori
        images_features = F.normalize(images_features, p=2, dim=-1)
        texts_features = F.normalize(texts_features, p=2, dim=-1)
        similarity_matrix = F.cosine_similarity(images_features.unsqueeze(1), texts_features.unsqueeze(0), dim=-1)

        num_images = similarity_matrix.shape[0]
        num_texts = similarity_matrix.shape[1]
        top_text_indices = np.argmax(similarity_matrix.cpu().numpy(), axis=1)

        # Calcola l'accuratezza
        correct = sum([1 if i == top_text_indices[i] else 0 for i in range(num_images)])
        correct_matches += correct
        # Calcola l'accuratezza di classe, cioè vediamo il report reale che classe è
        for i in range(num_images):
            true_label = labels[labels['study_id'] == study_ids[i]].iloc[:, 2:]
            fake_label = labels[labels['study_id'] == study_ids[top_text_indices[i][0]]].iloc[:, 2:]
            true_label = true_label.to_numpy().flatten()
            fake_label = fake_label.to_numpy().flatten()
            # se hanno gli 1 e gli 0 nello stesso posto allora sono uguali
            true_idx_1 = np.where(true_label == 1)[0]
            fake_idx_1 = np.where(fake_label == 1)[0]
            true_idx_0 = np.where(true_label == 0)[0]
            fake_idx_0 = np.where(fake_label == 0)[0]
            if np.array_equal(true_idx_1, fake_idx_1) and np.array_equal(true_idx_0, fake_idx_0):
                correct_class_matches += 1

        if top_text_indices[0][0] != 0:
            color = 'XXXXXX'
        else:
            color = 'VVVVVV'

        print(f"{color}--- True: {true_text[0]}")
        print(f"{color}--- Predicted: {texts[top_text_indices[0][0]]}")
    if j == n_test:
        break
accuracy = correct_matches / (num_images * n_test)
accuracy_class = correct_class_matches / (num_images * n_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Accuracy Class: {accuracy_class * 100:.2f}%")



