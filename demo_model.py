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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(torch.cuda.device_count())

model_load_paths = ['CoDi_encoders.pth', 'CoDi_text_diffuser.pth', 'CoDi_video_diffuser_8frames.pth']
inference_tester = dani_model(model='thesis_model', data_dir='checkpoints/', pth=model_load_paths, load_weights=True)  # turn on fp16=True if loading fp16 weights
"""
clip_weights = 'Clip_Training/saved_checkpoints/Training_Clip_5e^-5/checkpoint_29_epoch_Training_Clip_5e^-5.pt'
a, b = inference_tester.net.clip.load_state_dict(torch.load(clip_weights, map_location=device), strict=False)

optimus_weights = 'Report_Training/saved_checkpoints/VAE/checkpoint_29_epoch_VAE-Training-Prova1.pt'
optimus_weights = torch.load(optimus_weights, map_location='cpu')
a, b = inference_tester.net.optimus.load_state_dict(optimus_weights, strict=False)

frontal_weights = 'CXR_Training/saved_checkpoints/Frontal/checkpoint_99_epoch_New-Training-Frontal.pt'
frontal_weights = torch.load(frontal_weights, map_location='cpu')
for key in list(frontal_weights.keys()):  # Utilizza list per creare una copia delle chiavi
    if 'unet_image' in key:
        value = frontal_weights.pop(key)
        new_key = key.replace('unet_image', 'unet_frontal')
        frontal_weights[new_key] = value

lateral_weights = 'CXR_Training/saved_checkpoints/Lateral/checkpoint_40_epoch_New-Training-Lateral.pt'
lateral_weights = torch.load(lateral_weights, map_location='cpu')
for key in list(lateral_weights.keys()):  # Utilizza list per creare una copia delle chiavi
    if 'unet_image' in key:
        value = lateral_weights.pop(key)
        new_key = key.replace('unet_image', 'unet_lateral')
        lateral_weights[new_key] = value

a, b = inference_tester.net.model.load_state_dict(frontal_weights, strict=False)
a, b = inference_tester.net.model.load_state_dict(lateral_weights, strict=False)

text_weights = 'Report_Training/saved_checkpoints/Unet/checkpoint_0_epoch_Report_Diffusion_Training.pt'
text_weights = torch.load(text_weights, map_location='cpu')
a, b = inference_tester.net.model.load_state_dict(text_weights, strict=False)
"""
df1 = pd.read_csv('test_short_frontal_clean.csv')
df2 = pd.read_csv('test_short_lateral_clean.csv')
"""
prompt = df1['report'][0]

# 1) Passiamo il prompt al CLIP
ctx = inference_tester.net.clip_encode_text(1*[prompt], encode_type='encode_text').to(device)
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
    steps=50,
    shape=shapes,
    condition=conditioning,
    unconditional_guidance_scale=scale,
    xtype=['frontal'],
    condition_types=['text'],
    eta=1,
    verbose=False,
    mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

# adesso la passiamo al decoder
x = inference_tester.net.autokl_decode(z[0])

x = torch.clamp((x[0]+1.0)/2.0, min=0.0, max=1.0)
im = x[0].cpu().numpy()
im = exposure.equalize_hist(im)
"""
# facciamo un subplot con due immagini
path = df1['path'][0]
# aggiungiamo 256 davanti
path = '256/' + path
# sostituisco .dcim con .tiff
path = path.replace('.dcm', '.tiff')
im_real = tifffile.imread(path)
"""
fig, ax = plt.subplots(1, 2)
# mettiamo come titolo il prompt
fig.suptitle(prompt)
ax[0].imshow(im_real, cmap='gray')
ax[1].imshow(im, cmap='gray')

# salviamo il grafico nella cartella plots4
plt.savefig('plot1.png')
"""
# Ora invece usiamo im_real come conditioning per generare il testo
conditioning = []
im_real = torch.tensor(im_real, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
cim = inference_tester.net.clip_encode_vision(im_real, encode_type='encode_vision').to(device)
uim = None
scale = 2.0
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
x = inference_tester.net.optimus_decode(z[0])

x = [a.tolist() for a in x]
text = [inference_tester.net.optimus.tokenizer_decoder.decode(a) for a in x]

for i in range(len(text)):
    print('TEXT:', text[i])


prompt = df2['report'][0]

z, _ = inference_tester.sampler.sample(
    steps=50,
    shape=shapes,
    condition=conditioning,
    unconditional_guidance_scale=scale,
    xtype=['lateral'],
    condition_types=['text'],
    eta=1,
    verbose=False,
    mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

# adesso la passiamo al decoder
x = inference_tester.net.autokl_decode(z[0])

x = torch.clamp((x[0]+1.0)/2.0, min=0.0, max=1.0)
im = x[0].cpu().numpy()
im = exposure.equalize_hist(im)

# facciamo un subplot con due immagini
path = df2['path'][0]
# aggiungiamo 256 davanti
path = '256/' + path
# sostituisco .dcm con .tiff
path = path.replace('.dcm', '.tiff')
im_real = tifffile.imread(path)
fig, ax = plt.subplots(1, 2)
# mettiamo come titolo il prompt
fig.suptitle(prompt)
ax[0].imshow(im_real, cmap='gray')
ax[1].imshow(im, cmap='gray')

# salviamo il grafico nella cartella plots4
plt.savefig('plot2.png')

# Ora invece usiamo im_real come conditioning per generare il testo
conditioning = []
im_real = torch.tensor(im_real, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
cim = inference_tester.net.clip_encode_vision(im_real, encode_type='encode_vision').to(device)
uim = None
scale = 2.0
dummy = torch.zeros_like(im_real).to(device)
uim = inference_tester.net.clip_encode_vision(im_real, encode_type='encode_vision').to(device)
conditioning.append(torch.cat([uim, cim]))

shapes = []
n = 768
shape = [4, n]
shapes.append(shape)

z, _ = inference_tester.sampler.sample(
    steps=50,
    shape=shapes,
    condition=conditioning,
    unconditional_guidance_scale=scale,
    xtype=['text'],
    condition_types=['lateral'],
    eta=1,
    verbose=False,
    mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

# adesso la passiamo al decoder
x = inference_tester.net.optimus_decode(z)

x = [a.tolist() for a in x]
text = [inference_tester.net.optimus.tokenizer_decoder.decode(a) for a in x]

for i in range(len(text)):
    print('TEXT:', text[i])
