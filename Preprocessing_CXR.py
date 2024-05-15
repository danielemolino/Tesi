# In questo script, attuiamo tutto il preprocessing necessario per preparare il MIMIC-CXR.
# Gli step sono i seguenti:
# 1. Caricare una delle immagini del dataset, che sono in formato DICOM
# 2. Tiro fuori il pixel array
# 3. Controllare quanti bit sono utilizzati per rappresentare i pixel
#    Controllare il pixel spacing se è sempre lo stesso
#    Controllare anche il monochrome1 e monochrome2, in caso sia 1, invertire i pixel
# 4. Normalizzare l'immagine
# 5. Segmentare i polmoni dall'immagine (Tralasciato perchè vogliamo tenere anche le vista laterali)
# 6. Estrarre la bounding box dei polmoni
# 7. Ritagliare l'immagine per farla diventare quadrata (256x256) e (512x512)
# 8. Salvare l'immagine come tiff

# Come prima cosa, ho estratto il Pixel Spacing di tutte le immagini,
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pydicom
from torchvision.transforms.functional import resize
from PIL import Image
from scipy.ndimage import zoom
import imageio
import tifffile

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


df = pd.read_csv('Pixel_Spacing.csv')
spacing = df['field']
unique, counts = np.unique(spacing, return_counts=True)
plt.bar(unique, counts)
plt.xlabel('Pixel Spacing')
plt.ylabel('Counts')
plt.title('Pixel Spacing Distribution')
plt.show()

# Abbiamo visto che il valore più comune è [0.139, 0.139], quindi useremo questo valore per il preprocessing

# PREPROCESSING
root_dir = 'E:/CXR/physionet.org/files/mimic-cxr/2.0.0/files/'
dest_dir_256 = 'E:/CXR_256/'
dest_dir_512 = 'E:/CXR_512/'
csv = pd.read_csv('Mimic-CXR.csv')

# Estraiamo tutti quelli il cui subject_id inizia con 10
# trasformiamo il subject_id in stringa
csv['subject_id'] = csv['subject_id'].astype(str)
csv14 = csv[csv['subject_id'].str.startswith('16')]

for i in tqdm(range(len(csv14))):
    row = csv14.iloc[i]
    image_path = os.path.join(root_dir, row['path'])
    if not os.path.exists(image_path):
        continue
    path = 'p' + row['subject_id'][:2] + '/p' + row['subject_id']+ '/s' + row['study_id'].astype(str) + '/' + row['dicom_id']
    dest_path_256 = os.path.join(dest_dir_256, path)
    dest_path_512 = os.path.join(dest_dir_512, path)
    # Se i file esistono già, continua
    if os.path.exists(dest_path_256 + '.tiff') and os.path.exists(dest_path_512 + '.tiff'):
        continue
    dicom = pydicom.dcmread(image_path)
    image = dicom.pixel_array
    # Controllo Monochrome1 e Monochrome2
    if dicom.PhotometricInterpretation == 'MONOCHROME1':
        image = (2 ** dicom.BitsStored - 1) - image
    # Controllo Pixel Spacing
    if dicom.ImagerPixelSpacing != [0.139, 0.139]:
        zoom_factor = [0.139 / dicom.ImagerPixelSpacing[0], 0.139 / dicom.ImagerPixelSpacing[1]]
        image = zoom(image, zoom_factor)
    # Normalizzazione
    image = image / (2 ** dicom.BitsStored - 1)
    # Se l'immagine non è quadrata, facciamo padding
    if image.shape[0] != image.shape[1]:
        diff = abs(image.shape[0] - image.shape[1])
        pad_size = diff // 2
        if image.shape[0] > image.shape[1]:
            padded_image = np.pad(image, ((0, 0), (pad_size, pad_size)))
        else:
            padded_image = np.pad(image, ((pad_size, pad_size), (0, 0)))
    # Resizing a 256x256 e a 512x512
    zoom_factor = [256 / padded_image.shape[0], 256 / padded_image.shape[1]]
    image_256 = zoom(padded_image, zoom_factor)
    zoom_factor = [512 / padded_image.shape[0], 512 / padded_image.shape[1]]
    image_512 = zoom(padded_image, zoom_factor)
    # Salviamo le immagini
    # se le cartelle non esistono, creale
    if not os.path.exists(os.path.dirname(dest_path_256)):
        os.makedirs(os.path.dirname(dest_path_256))
    if not os.path.exists(os.path.dirname(dest_path_512)):
        os.makedirs(os.path.dirname(dest_path_512))
    tifffile.imwrite(dest_path_256 + '.tiff', image_256)
    tifffile.imwrite(dest_path_512 + '.tiff', image_512)
