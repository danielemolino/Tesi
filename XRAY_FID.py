# In questo script, usiamo l'implementazione del FID da https://github.com/mseitzer/pytorch-fid
import torchxrayvision as xrv
import skimage, torchvision
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from core.models.model_module_infer import model_module
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


def generation(device, dataloader, unet_weights):
    model_load_paths = ['CoDi_encoders.pth', 'CoDi_video_diffuser_8frames.pth']
    inference_tester = model_module(data_dir='checkpoints/', pth=model_load_paths,
                                    fp16=False, load_weights=False)  # turn on fp16=True if loading fp16 weights
    inference_tester = inference_tester.to(device)
    inference_tester.eval()
    # prendiamo tutti i .pt nella cartella saved_checkpoints con os.listddir
    #clip_weights = 'Clip_Training/saved_checkpoints/DirtyData-20Ep-5e^-5/checkpoint_19_epoch_DirtyData-20Ep-5e^-5.pt'
    #a, b = inference_tester.net.clip.load_state_dict(torch.load(clip_weights, map_location=device), strict=False)

    #a, b = inference_tester.net.model.load_state_dict(torch.load(unet_weights, map_location=device), strict=False)
    print(f'++++++ {unet_weights} ++++++')
    j = 0
    axis_dict = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (0, 5), 6: (0, 6), 7: (0, 7), 8: (1, 0),
                 9: (1, 1), 10: (1, 2), 11: (1, 3), 12: (1, 4), 13: (1, 5), 14: (1, 6), 15: (1, 7), 16: (2, 0),
                 17: (2, 1), 18: (2, 2), 19: (2, 3), 20: (2, 4), 21: (2, 5), 22: (2, 6), 23: (2, 7), 24: (3, 0),
                 25: (3, 1), 26: (3, 2), 27: (3, 3), 28: (3, 4), 29: (3, 5), 30: (3, 6), 31: (3, 7)}

    # creiamo un csv con i path delle immagini generate, creiamo un dataframe vuoto
    df = pd.DataFrame(columns=['path'])
    numbers = extract_numbers(unet_weights)[0]
    numbers = list(map(int, extract_numbers(unet_weights)))[0]
    for batch in dataloader:
        _, prompt, subject_id, study_id, path = batch
        n_samples = len(prompt)
        prompt = list(prompt)
        input_texts= inference_tester.net.clip.tokenizer(prompt, truncation=True, max_length=inference_tester.net.clip.max_length, return_length=True,
                                      return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        input_texts = input_texts["input_ids"]
        ctx = inference_tester.net.clip(input_texts, 'encode_text')
        utx = None
        scale = 2.0
        conditioning = []
        if scale != 1.0:
            utx = inference_tester.net.clip_encode_text(n_samples * [""], encode_type='encode_text').to(device)
        conditioning.append(torch.cat([utx, ctx]))

        h, w = [256, 256]
        shapes = []
        shape = [n_samples, 4, h // 8, w // 8]
        shapes.append(shape)

        z, _ = inference_tester.sampler.sample(
            steps=1,
            shape=shapes,
            condition=conditioning,
            unconditional_guidance_scale=scale,
            xtype=['image'],
            condition_types=['text'],
            eta=1,
            verbose=False,
            mix_weight={'video': 1, 'audio': 1, 'text': 1, 'image': 1})

        image = inference_tester.net.autokl_decode(z[0])

        for i in range(n_samples):
            im = image[i]
            path = path[i]
            x = torch.clamp((im[0] + 1.0) / 2.0, min=0.0, max=1.0)
            im = x[0].cpu().numpy()
            dest_path_256 = f'256_generated/{path}'
            dest_path_256 = dest_path_256.replace('.dcm', '.tiff')
            df = df._append({'path': dest_path_256}, ignore_index=True)
            if not os.path.exists(os.path.dirname(dest_path_256)):
                os.makedirs(os.path.dirname(dest_path_256))
            # al posto di .dcm mettiamo .tiff
            tifffile.imwrite(dest_path_256, im)
    # Adesso, una volta che abbiamo generato tutte le immagini, possiamo salvare il dataframe in un csv
    df.to_csv('256_generated.csv', index=False)
    df_to_list = df['path'].tolist()
    return df_to_list


IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        im = tifffile.imread(path)
        im = im * 255
        im = xrv.datasets.normalize(im, 255)  # convert 8-bit image to [-1024, 1024] range
        im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        im = im.mean(2)[None, ...]  # Make single color channel
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
        im = transform(im)
        im = torch.from_numpy(im)
        return im


def get_activations(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = len(files)

    dataset = ImagePathDataset(files)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            _, pred = model(batch)  # or model.features(img[None,...])

        pred = pred.cpu().numpy()

        pred_arr[start_idx: start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
            mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
            sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
                  "fid calculation produces singular product; "
                  "adding %s to diagonal of cov estimates"
              ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
        files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=1):
    files = path
    m, s = calculate_activation_statistics(
        files, model, batch_size, dims, device, num_workers
    )
    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1, weights='densenet121-res224-mimic_ch'):
    """Calculates the FID of two paths"""
    for p in paths:
        for file in p:
            if not os.path.exists(file):
                pass

    model = xrv.models.DenseNet(weights=weights).to(device)

    m1, s1 = compute_statistics_of_path(
        paths[0], model, batch_size, dims, device, num_workers
    )
    m2, s2 = compute_statistics_of_path(
        paths[1], model, batch_size, dims, device, num_workers
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    num_workers = 4
    save_stats = False
    batch_size = 50
    dims = 1024  # Dimensionality of Inception features to use
    to_generate = True
    n_samples = 1000
    view = 'Frontal'
    xrv_weights = ['densenet121-res224-mimic_ch', 'densenet121-res224-all']

    """unet_weights = os.listdir(f'CXR_Training/saved_checkpoints/{view}')
    unet_weights = [f'CXR_Training/saved_checkpoints/{view}/{x}' for x in unet_weights if x.endswith('.pt')]
    # Controlliamo se già esiste il file 	CXR_Frontal_FID_XRV.csv
    if os.path.exists('csv/CXR_{view}_FID_XRV.csv'):
        # leggiamo i weights già presenti
        df = pd.read_csv('csv/CXR_{view}_FID_XRV.csv')
        weights = df['Weight'].tolist()
        # eliminiamo da unet_weights i weights già presenti
        unet_weights = [x for x in unet_weights if x not in weights]"""
    unet_weights = ['CXR_Training/saved_checkpoints/Frontal/epoch_0.pt']

    for w in unet_weights:
        if to_generate:
            view2 = view.lower()
            csv = pd.read_csv(f'test_short_{view2}_clean.csv')
            dataset = MIMIC_CXR_Dataset(csv, '256/')
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
            path1 = generation(device, dataloader, w)
        else:
            path1 = pd.read_csv('256_generated.csv')
            path1 = path1['path'].tolist()

        # sostituiamo in path1 .dcm con .tiff
        path1 = [x.replace('.dcm', '.tiff') for x in path1]
        # teniamo le prime 50
        path2 = [x.replace('256_generated/', '256/') for x in path1]
        path2 = path1

        path = [path1, path2]
        for xrwv in xrv_weights:
            fid_value = calculate_fid_given_paths(path, batch_size, device, dims, num_workers, xrwv)
            print("FID: ", fid_value)
            # salviamo anche il valore del FID in un file CSV
            output_path = f'csv/CXR_{view}_FID_XRV.csv'
            # Se non esiste, creiamolo, deve avere le colonne Weight e FID
            if not os.path.exists(output_path):
                df = pd.DataFrame(columns=['XRV', 'Weight', 'FID'])
                df.to_csv(output_path, index=False)
            df = pd.read_csv(output_path)
            df = df._append({'XRV': xrvw, 'Weight': view + w, 'FID': fid_value}, ignore_index=True)
            df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
