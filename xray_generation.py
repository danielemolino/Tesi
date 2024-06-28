# In questo script, usiamo l'implementazione del FID da https://github.com/mseitzer/pytorch-fid
import torchxrayvision as xrv
import skimage, torchvision
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from core.models.model_module_infer import model_module
from core.models.dani_model import dani_model
import matplotlib.pyplot as plt
import tifffile
import pathlib
import numpy as np
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from Clip_Training.DataLoader import MIMIC_CXR_Dataset
import re


from pytorch_fid.inception import InceptionV3
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def extract_numbers(string):
    return re.findall(r'\d+', string)


def generation(device, dataloader, unet_weights, numbers, target, source):
    target = target.lower()
    source = source.lower()
    model_load_paths = ['CoDi_encoders.pth', 'CoDi_video_diffuser_8frames.pth']
    inference_tester = dani_model(model='thesis_model', data_dir='checkpoints/', pth=model_load_paths,
                                    fp16=False, load_weights=True)  # turn on fp16=True if loading fp16 weights
    inference_tester = inference_tester.to(device)
    inference_tester.eval()
    # prendiamo tutti i .pt nella cartella saved_checkpoints con os.listddir
    clip_weights = 'Clip_Training/saved_checkpoints/Training_Clip_5e^-5/checkpoint_29_epoch_Training_Clip_5e^-5.pt'
    a, b = inference_tester.net.clip.load_state_dict(torch.load(clip_weights, map_location=device), strict=False)

    a, b = inference_tester.net.model.load_state_dict(torch.load(unet_weights, map_location=device), strict=False)
    print(f'++++++ {numbers} ++++++')

    # creiamo un csv con i path delle immagini generate, creiamo un dataframe vuoto
    df = pd.DataFrame(columns=['path'])
    # per ogni batch con tqdm
    for batch in tqdm(dataloader):
        with torch.no_grad():
            images, _, subject_id, study_id, paths = batch
            images = images.unsqueeze(1).float()
            paths = list(paths)
            n_samples = len(prompt)

            conditioning = []
            cim = inference_tester.net.clip(images.to(device), encode_type='encode_vision').to(device)
            uim = None
            scale = 7.5
            dummy = torch.zeros_like(images).to(device)
            uim = inference_tester.net.clip_encode_vision(dummy, encode_type='encode_vision').to(device)
            conditioning.append(torch.cat([uim, cim]))

            h, w = [256, 256]
            shapes = []
            shape = [n_samples, 4, h // 8, w // 8]
            shapes.append(shape)

            z, _ = inference_tester.sampler.sample(
                steps=50,
                shape=shapes,
                condition=conditioning,
                unconditional_guidance_scale=scale,
                xtype=[target],
                condition_types=[source],
                eta=1,
                verbose=False,
                mix_weight={'video': 1, 'audio': 1, 'text': 1, 'image': 1})

            image = inference_tester.net.autokl_decode(z[0])
            print(image.shape)

            for i in range(n_samples):
                im = image[i]
                path = paths[i]
                x = torch.clamp((im[0] + 1.0) / 2.0, min=0.0, max=1.0)
                im = x.cpu().numpy()
                dest_path_256 = f'256_generated_{numbers}_{source}_to_{target}/{path}'
                dest_path_256 = dest_path_256.replace('.dcm', '.tiff')
                df = df._append({'path': dest_path_256}, ignore_index=True)
                if not os.path.exists(os.path.dirname(dest_path_256)):
                    os.makedirs(os.path.dirname(dest_path_256))
                # al posto di .dcm mettiamo .tiff
                tifffile.imwrite(dest_path_256, im)

        # Adesso, una volta che abbiamo generato tutte le immagini, possiamo salvare il dataframe in un csv
        df.to_csv(f'csv/256_generated_{numbers}_{source}_to_{target}.csv', index=False)
        df_to_list = df['path'].tolist()
        return df_to_list


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    num_workers = 4
    batch_size = 50
    view_target = 'Frontal'
    source = 'Lateral'

    unet_weights = os.listdir(f'/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/CXR_Training/saved_checkpoints/{view_target}')
    unet_weights = [f'/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/CXR_Training/saved_checkpoints/{view_target}/{x}' for x in unet_weights if x.endswith('.pt')]
    # Controlliamo se già esiste il file 	CXR_Frontal_FID_XRV.csv
    unet_weights = [unet_weights[0]]
    for w in unet_weights:
        print(f"Using weights: {w}")
        view2 = source.lower()
        csv = pd.read_csv(f'csv/test_short_{view2}_clean.csv')
        dataset = MIMIC_CXR_Dataset(csv, '256/')
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        numbers = extract_numbers(w)
        path1 = generation(device, dataloader, w, numbers[-1], view_target, source)


if __name__ == '__main__':
    main()


