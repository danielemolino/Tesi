import os
from core.models.dani_model import dani_model
from PIL import Image
import torch
import torch.nn as nn
from core.common.utils import remove_duplicate_word
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage import exposure
import pandas as pd
from tqdm import tqdm


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(torch.cuda.device_count())

model_load_paths = ['CoDi_encoders.pth', 'CoDi_text_diffuser.pth', 'CoDi_video_diffuser_8frames.pth']
inference_tester = dani_model(model='thesis_model', data_dir='checkpoints/', pth=model_load_paths, load_weights=False)

clip_weights = 'Clip_Training/saved_checkpoints/Training_Clip_5e^-5/checkpoint_29_epoch_Training_Clip_5e^-5.pt'
a, b = inference_tester.net.clip.load_state_dict(torch.load(clip_weights, map_location=device), strict=False)

inference_tester = inference_tester.to(device)

df = pd.read_csv('csv/train_short_clean.csv')

# Adesso, per ciascuna riga del dataframe, dobbiamo estrarre l'embedding sia dell'immagine che del testo, ma essendo uno stesso testo associato a più immagini, dobbiamo fare in modo che il testo venga estratto una sola volta
# Un idea potrebbe essere quella di salvare tutti gli embedding frontali in un numpy array, tutti gli embedding testuali in un altro numpy array e tutti i laterali in un altro numpy array
# Nei vari npy che salvo, voglio avere, study_id e embedding in quello testuale, study_id, dicom_id e embedding in quello frontale e study_id, dicom_id e embedding in quello laterale

# Iniziamo creando i tre numpy array
text_embeddings = []
frontal_embeddings = []
lateral_embeddings = []

# Definiamo un dtype personalizzato per i numpy array
dtype_txt = np.dtype([
    ('study_id', 'int64'),
    ('embedding', 'float32', (768,))
])

dtype_im = np.dtype([
    ('study_id', 'int64'),        # Intero a 8 cifre (int64 permette di memorizzare numeri grandi a sufficienza)
    ('dicom_id', 'U44'),           # Stringa di lunghezza massima 10 caratteri (puoi aumentare se necessario)
    ('embedding', 'float32', (768,))  # Vettore di 768 float32
])

for i in tqdm(range(len(df))):
    report = df.iloc[i]['report']
    study_id = df.iloc[i]['study_id']
    dicom_id = df.iloc[i]['dicom_id']
    view_position = df.iloc[i]['ViewPosition']
    path = '256/' + df.iloc[i]['path']
    # Sostituiamo .dcm con .tiff
    path = path.replace('.dcm', '.tiff')
    # Carichiamo l'immagine
    im = tifffile.imread(path)
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Passiamo il report al clip
    with torch.no_grad():
        text_embedding = inference_tester.net.clip_encode_text([report]).squeeze(0).cpu().numpy()
        im_embedding = inference_tester.net.clip_encode_vision(im).cpu().numpy()

    text_embeddings.append((study_id, text_embedding))

    if view_position == 'AP' or view_position == 'PA':
        frontal_embeddings.append((study_id, dicom_id, im_embedding))
    else:
        lateral_embeddings.append((study_id, dicom_id, im_embedding))

text_embeddings = np.array(text_embeddings, dtype=dtype_txt)
frontal_embeddings = np.array(frontal_embeddings, dtype=dtype_im)
lateral_embeddings = np.array(lateral_embeddings, dtype=dtype_im)

np.save('text_embeddings.npy', text_embeddings)
np.save('frontal_embeddings.npy', frontal_embeddings)
np.save('embeddings/lateral_embeddings.npy', lateral_embeddings)

text_embeddings = np.load('text_embeddings.npy', allow_pickle=True)
frontal_embeddings = np.load('frontal_embeddings.npy', allow_pickle=True)



