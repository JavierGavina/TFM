import matplotlib.pyplot as plt
import numpy as np
from src.constants import constants
from src.dataloader import labelDecoding
import pandas as pd
from models.gans_utils import DataAugmentation, get_path_cgan, get_path_wcgan_gp, get_path_wcgan
import tensorflow as tf
from src.dataloader import DataLoader
import warnings
from scipy.stats import entropy
from scipy.linalg import sqrtm
import os
from keras.applications.inception_v3 import InceptionV3
from skimage.transform import resize
# clear_output
from IPython.display import clear_output

warnings.filterwarnings("ignore")


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid



def kl_divergence(reals: np.ndarray, fakes: np.ndarray) -> np.float32:
    """
    Métrica que mide cuanta información estamos perdiendo cuando utilizamos la distribución de fakes (Q) para aproximar la distribución de reals (P)

    Parameters
    ----------
    reals: np.ndarray
        Imagen real
    fakes: np.ndarray
        Imagen sintética

    Returns
    -------
    kl_metric: np.float
        Valor de la divergencia de Kullback-Leibler
    """

    # Clip values to avoid division by zero
    reals = np.clip(reals, 1e-10, None)
    fakes = np.clip(fakes, 1e-10, None)

    # Convert to probabilities
    reals /= np.sum(reals)
    fakes /= np.sum(fakes)

    kl_metric = entropy(reals.flatten(), fakes.flatten())
    return kl_metric


# prepare the inception v3 model
def InceptionScore(reals, fake):
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(100, 100, 3))
    reals = scale_images(reals, (100, 100, 3))
    fake = scale_images(np.expand_dims(fake, axis=-1), (100, 100, 3))

    # calculate fid
    fid = calculate_fid(model, reals, fake)
    return fid


if __name__ == "__main__":
    os.makedirs(f"../{constants.outputs.METRICS}", exist_ok=True)
    # Cargamos los datos reales
    get_loader = DataLoader(
        data_dir=f"{constants.data.train.FINAL_PATH}/groundtruth.csv",
        aps_list=constants.aps, batch_size=30, step_size=5,
        size_reference_point_map=28, return_axis_coords=False
    )

    X, y, _ = get_loader()

    # Guardamos los parámetros de todos los modelos entrenados de cada tipo de GAN
    params_wcgan = {
        "model": "WCGAN",
        "learning_rate": [0.00005, 0.0001, 0.0005, 0.001, 0.005],
        "n_critic": [1, 2, 3, 4, 5],
        "clip_value": [0.001, 0.005, 0.01, 0.05, 0.1]
    }

    params_wcgan_gp = {
        "model": "WCGAN-GP",
        "learning_rate": [0.00005, 0.0001, 0.0005, 0.001, 0.005],
        "n_critic": [1, 2, 3, 4, 5],
        "gradient_penalty": [5.0, 10.0, 15.0]
    }

    params_cgan = {
        "model": "CGAN",
        "learning_rate": [0.00005, 0.0001, 0.0005, 0.001, 0.005]
    }
    nAPs = len(constants.aps)
    samples_per_ap = X.shape[0] // nAPs

    count = 0
    n_total = len(params_wcgan_gp["learning_rate"]) * len(params_wcgan_gp["n_critic"]) * len(params_wcgan_gp["gradient_penalty"]) * len(constants.aps) + \
              len(params_wcgan["learning_rate"]) * len(params_wcgan["n_critic"]) * len(params_wcgan["clip_value"]) * len(constants.aps) + \
              len(params_cgan["learning_rate"]) * len(constants.aps)
    df = pd.DataFrame(columns=["Model", "Label", "Learning Rate", "N Critic", "Clip Value", "Gradient Penalty",
                               "Kullback-Leibler Divergence", "Inception Score"])

    # Calculamos las métricas para cada modelo

    # CGAN
    for lr in params_cgan["learning_rate"]:
        path = f"../{get_path_cgan(learning_rate=lr)}"
        if os.path.exists(path):
            x_cgan, _ = DataAugmentation(path_to_generator=path)(n_samples_per_label=samples_per_ap)
        for x in range(0, samples_per_ap * nAPs, samples_per_ap):
            count += 1
            fake_mean = x_cgan[x:x + samples_per_ap].mean(axis=0)
            real_mean = X[x:x + samples_per_ap, :, :, 0].mean(axis=0)
            kl_div = kl_divergence(reals=real_mean, fakes=fake_mean)
            in_score = InceptionScore(reals=X[x:x + samples_per_ap, :, :, 0], fake=x_cgan[x:x + samples_per_ap])
            df.loc[df.shape[0]] = ["CGAN", constants.aps[int(x / samples_per_ap)], lr, np.nan, np.nan, np.nan, kl_div, in_score]
            print("CGAN", "PROGRESS", count, "/", n_total, "INCEPTION", in_score)
            if count % 10 == 0:
                clear_output()
    # WCGAN
    for lr in params_wcgan["learning_rate"]:
        for nc in params_wcgan["n_critic"]:
            for cv in params_wcgan["clip_value"]:
                path = f"../{get_path_wcgan(learning_rate=lr, n_critic=nc, clip_value=cv)}"
                if os.path.exists(path):
                    x_wcgan, _ = DataAugmentation(path_to_generator=path)(n_samples_per_label=samples_per_ap)
                    for x in range(0, samples_per_ap * nAPs, samples_per_ap):
                        count += 1
                        fake_mean = x_wcgan[x:x + samples_per_ap].mean(axis=0)
                        real_mean = X[x:x + samples_per_ap, :, :, 0].mean(axis=0)
                        kl_div = kl_divergence(reals=real_mean, fakes=fake_mean)
                        in_score = InceptionScore(reals=X[x:x + samples_per_ap, :, :, 0], fake=x_wcgan[x:x + samples_per_ap])
                        df.loc[df.shape[0]] = ["WCGAN", constants.aps[int(x / samples_per_ap)], lr, nc, cv, np.nan, kl_div, in_score]
                        print("WCGAN", "PROGRESS", count, "/", n_total, "INCEPTION", in_score)
                        if count % 10 == 0:
                            clear_output()

    # WCGAN-GP
    for lr in params_wcgan_gp["learning_rate"]:
        for nc in params_wcgan_gp["n_critic"]:
            for gp in params_wcgan_gp["gradient_penalty"]:
                path = f"../{get_path_wcgan_gp(learning_rate=lr, n_critic=nc, gradient_penalty=gp)}"
                if os.path.exists(path):
                    x_wcgan_gp, _ = DataAugmentation(path_to_generator=path)(n_samples_per_label=samples_per_ap)
                    for x in range(0, samples_per_ap * nAPs, samples_per_ap):
                        count += 1
                        print("WCGAN_GP", "PROGRESS", count, "/", n_total)
                        fake_mean = x_wcgan_gp[x:x + samples_per_ap].mean(axis=0)
                        real_mean = X[x:x + samples_per_ap, :, :, 0].mean(axis=0)
                        kl_div = kl_divergence(reals=real_mean, fakes=fake_mean)
                        in_score = InceptionScore(reals=X[x:x + samples_per_ap, :, :, 0], fake=x_wcgan_gp[x:x + samples_per_ap])
                        df.loc[df.shape[0]] = ["WCGAN-GP", constants.aps[int(x / samples_per_ap)], lr, nc, np.nan, gp, kl_div,
                                               in_score]
                        print("WCGAN_GP", "PROGRESS", count, "/", n_total, "INCEPTION", in_score)
                        if count % 10 == 0:
                            clear_output()


# Exportamos los resultados a un csv
df.to_csv(f"../{constants.outputs.METRICS}/metrics.csv", index=False)
