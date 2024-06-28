import os
import torchxrayvision as xrv
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import pandas as pd
import skimage, torch, torchvision
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and process image
model = xrv.models.DenseNet(weights="densenet121-res224-all")

model = model.to(device)

transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
data_aug = None

test_options = {'codi': ("/256", '_codi', 'tiff'), 'test': ("../TestSet_Mimic", '', 'jpg')}
to_test = 'codi'

dataset = xrv.datasets.MIMIC_Dataset(
    imgpath=test_options[to_test][0],
    csvpath=f"labels_test{test_options[to_test][1]}.csv",
    metacsvpath=f"metadata_test{test_options[to_test][1]}.csv",
    transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA", "AP"], dtype=test_options[to_test][2])
# Eliminiamo da dataset.pathologies le colonne che non ci interessano, cioè Pleural Other, Fracture e Support Devices

dl = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
# Creiamo un dict che collega l'ordine delle colonne di results con l'ordine dell'output del modello

# Con tqdn
for batch in tqdm(dl):
    with torch.no_grad():
        data = batch

        im = data['img'].to(device)
        labels = data['lab']

        outputs = model(im)
        pred = outputs[0].cpu().detach().numpy()

        if 'results' not in locals():
            results = pd.DataFrame(pred, columns=model.pathologies)
        else:
            results = pd.concat([results, pd.DataFrame(pred, columns=model.pathologies)])

        if 'all_labels' not in locals():
            all_labels = pd.DataFrame(labels.numpy(), columns=dataset.pathologies)
        else:
            all_labels = pd.concat([all_labels, pd.DataFrame(labels.numpy(), columns=dataset.pathologies)])


# Una volta ottenuti i risultati, possiamo calcolare le metriche, voglio calcolare:
# - AUC
# - F1
dataset.pathologies.remove('Pleural Other')
dataset.pathologies.remove('Fracture')
dataset.pathologies.remove('Support Devices')

auc_dict = {disease: 0 for disease in dataset.pathologies}
f1_dict = {disease: 0 for disease in dataset.pathologies}

for disease in dataset.pathologies:
    y = all_labels[disease]
    x = results[disease]
    # Uniamole in due colonne e droppiamo tutte le righe con NaN
    y = y.reset_index(drop=True)
    x = x.reset_index(drop=True)
    df = pd.concat([y, x], axis=1)
    df = df.dropna()
    y = df.iloc[:, 0]
    x = df.iloc[:, 1]
    auc_dict[disease] = roc_auc_score(y, x)
    f1_dict[disease] = f1_score(y, x > 0.5)

print(auc_dict)
print(f1_dict)


