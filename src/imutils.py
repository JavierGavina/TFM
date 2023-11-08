import os

import PIL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import IPython
from scipy.interpolate import griddata
import pandas as pd


def get_concat_v(im1: PIL.PngImagePlugin.PngImageFile,
                 im2: PIL.PngImagePlugin.PngImageFile) -> PIL.PngImagePlugin.PngImageFile:
    """
    Esta función se encarga de concatenar dos imágenes verticalmente

    Parameters:
    ----------
    im1: PIL.PngImagePlugin.PngImageFile
        Imágenes que corresponden con las imágenes reales
    im2: PIL.PngImagePlugin.PngImageFile
        Imágenes que corresponden con las imágenes fake

    Returns:
    ----------
    dst : PIL.PngImagePlugin.PngImageFile
        Imágen PIL concatenada verticalmente de las imágenes reales y fake

    Examples
    --------

    Concatenar imagenes reales y falsas

    >>> real = Image.open("temps/real_0001.png")

    >>> fake = Image.open("temps/fake_0001.png")

    >>> concat = get_concat_v(real, fake)
    """

    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def plot_real_vs_fake(x_combined_batch: np.array, epoch: int, batch: int) -> None:
    """
    Esta función se encarga de mostrar las imágenes reales y las generadas por el modelo en cada época de la GAN

    Parameters:
    ----------
    x_combined_batch: np.ndarray
        batch de imágenes reales y generadas
    epoch: int
        época de la GAN
    batch: int
        Tamaño del lote

    Returns:
    ----------
    plot real vs fake images : none
        Muestra las imágenes reales y las generadas por el modelo en cada época de la GAN y las guarda en el directorio fingerprints_generated

    Examples
    --------

    Batch combinado de imágenes reales y generadas

    >>> x_combined_batch = np.concatenate((legit_images, syntetic_images))

    Visualizar las imágenes reales y generadas

    >>> plot_real_vs_fake(x_combined_batch, epoch, batch)
    """

    # Create a folder to save the images
    os.makedirs("temps", exist_ok=True)

    # Plot the images
    for idx in range(np.int64(batch / 2) * 2):

        # Plot the real images
        if idx < np.int64(batch / 2):
            plt.subplot(int(np.ceil(np.int64(batch / 2) / 4)), 4, idx + 1)
            plt.imshow(np.repeat(x_combined_batch[idx, :, :, 0], 4, axis=1), plt.cm.Greys)
            plt.axis('off')
            plt.title(f"reales {epoch}")

        # Save the real images
        if idx == np.int64(batch / 2) - 1:
            plt.tight_layout()
            plt.savefig('temps/real_{:04d}.png'.format(epoch))

        # Plot the fake images
        if idx >= np.int64(batch / 2):
            plt.subplot(int(np.ceil(np.int64(batch / 2) / 4)), 4, idx - (np.int64(batch / 2) - 1))
            plt.imshow(np.repeat(x_combined_batch[idx, :, :, 0], 4, axis=1), plt.cm.Greys)
            plt.axis('off')
            plt.title(f"fake {epoch}")

        # Save the fake images
        if idx == (np.int64(batch / 2) * 2) - 1:
            plt.subplot(int(np.ceil(np.int64(batch / 2) / 4)), 4, idx - (np.int64(batch / 2) - 1))
            plt.imshow(np.repeat(x_combined_batch[idx, :, :, 0], 4, axis=1), plt.cm.Greys)
            plt.axis('off')
            plt.title(f"fake {epoch}")
            plt.tight_layout()
            plt.savefig('temps/fake_{:04d}.png'.format(epoch))

    # Concatenate the real and fake images
    real = Image.open("temps/real_{:04d}.png".format(epoch))
    fake = Image.open("temps/fake_{:04d}.png".format(epoch))
    concat = get_concat_v(real, fake)

    # Save the concatenated images
    concat.save("fingerprints_generated/image_at_epoch_{:04d}.png".format(epoch))


def plot_real_vs_fake_conditional(x_combined_batch: np.ndarray, labels: np.ndarray, epoch: int, batch: int) -> None:
    """
    Esta función se encarga de mostrar las imágenes reales y las generadas por el modelo en cada época de la GAN condicional

    Parameters:
    ----------
    x_combined_batch: np.ndarray
        batch de imágenes reales y generadas
    labels: np.ndarray
        Etiquetas de las imágenes
    epoch: int
        época de la GAN
    batch: int
        Tamaño del lote
    """
    # Create a folder to save the images
    os.makedirs("temps", exist_ok=True)

    # Plot the images
    for idx in range(np.int64(batch / 2) * 2):

        # Plot the real images
        if idx < np.int64(batch / 2):
            plt.subplot(int(np.ceil(np.int64(batch / 2) / 4)), 4, idx + 1)
            plt.imshow(np.repeat(x_combined_batch[idx, :, :, 0], 4, axis=1), plt.cm.Greys)
            plt.axis('off')
            plt.title(f"reales ep: {epoch}, l: {labels.argmax(axis=1)[idx]}")

        # Save the real images
        if idx == np.int64(batch / 2) - 1:
            plt.tight_layout()
            plt.savefig('temps/real_{:04d}.png'.format(epoch))

        # Plot the fake images
        if idx >= np.int64(batch / 2):
            idx_label = idx - np.int64(batch / 2)
            plt.subplot(int(np.ceil(np.int64(batch / 2) / 4)), 4, idx_label + 1)
            plt.imshow(np.repeat(x_combined_batch[idx, :, :, 0], 4, axis=1), plt.cm.Greys)
            plt.axis('off')
            plt.title(f"fake ep: {epoch}, l: {labels.argmax(axis=1)[idx_label]}")

        # Save the fake images
        if idx == (np.int64(batch / 2) * 2) - 1:
            plt.subplot(int(np.ceil(np.int64(batch / 2) / 4)), 4, idx_label + 1)
            plt.imshow(np.repeat(x_combined_batch[idx, :, :, 0], 4, axis=1), plt.cm.Greys)
            plt.axis('off')
            plt.title(f"fake ep: {epoch}, l: {labels.argmax(axis=1)[idx_label]}")
            plt.tight_layout()
            plt.savefig('temps/fake_{:04d}.png'.format(epoch))

    # Concatenate the real and fake images
    real = Image.open("temps/real_{:04d}.png".format(epoch))
    fake = Image.open("temps/fake_{:04d}.png".format(epoch))
    concat = get_concat_v(real, fake)

    # Save the concatenated images
    concat.save("fingerprints_generated_conditional/image_at_epoch_{:04d}.png".format(epoch))


def obtainEvolutionGAN(folder_path: str = "fingerprints_generated", out_dir: str = "output.gif") -> None:
    """
    Esta función se encarga de generar un GIF con la evolución de la GAN

    :type out_dir: str
    :type folder_path: str

    :param folder_path: por defecto "fingerprints_generated". Dirección donde se encuentran las imágenes generadas en cada época
    :param out_dir: por defecto "output.gif". Dirección donde se guardará el GIF
    :return:  none
    """
    # List the image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    images = []

    # Open and append each image to the list
    for image_file in image_files:
        image = Image.open(os.path.join(folder_path, image_file))
        images.append(image)

    # Save the GIF
    images[0].save(out_dir, save_all=True, append_images=images[1:], duration=400, loop=0)

    print(f'GIF saved to {out_dir}')


def displayGIF(path: str) -> IPython.display.Image:
    """
    Esta función se encarga de mostrar un GIF
    :param path: Dirección del GIF
    :return: Muestra el GIF
    """
    return IPython.display.Image(url=path)


def interpolateAPImage(df: pd.DataFrame, ap: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpola la imagen de un AP en función de la latitud y longitud de cada Reference Point (RP)

    Parameters:
    ----------
    df: pd.DataFrame
        Dataframe con las columnas Longitude, Latitude y los APs
    ap: str
        nombre del AP (Punto de acceso o wifi)

    Returns:
    ----------
    x_g : ndarray
        Array ordenado con las longitudes de los RP
    y_g : ndarray
        Array ordenado con las latitudes de los RP.
    grid_z0 : ndarray
        Array con los valores RSSI de la imagen interpolada del AP

    Examples
    --------
    Leer los datos del ground truth
    >>> import pandas as pd

    >>> df = pd.read_csv("../data/final_groundtruth/groundtruth.csv")

    Seleccionar las columnas del WiFi

    >>> wifi_train = df[df.columns[:8].tolist()]

    Interpolar la imagen del AP 480Invitados para todos los (latitud, longitud) posibles

    >>> x_g, y_g, grid_z0 = interpolateAPImage(wifi_train, "480Invitados")

    Visualizar la imagen interpolada

    >>> plotInterpolatedAPImage(x_g, y_g, grid_z0, "480Invitados")
    """

    aux = df.groupby(["Longitude", "Latitude"]).mean()[ap].reset_index()
    miny, maxy = min(aux['Latitude']), max(aux['Latitude'])
    minx, maxx = min(aux['Longitude']), max(aux['Longitude'])
    grdi_x = np.linspace(minx, maxx, num=300, endpoint=False)
    grdi_y = np.linspace(miny, maxy, num=300, endpoint=False)
    yg, xg = np.meshgrid(grdi_y, grdi_x, indexing='ij')
    x_g = xg.ravel()
    y_g = yg.ravel()

    aux2 = aux.drop([ap], 1)
    aux3 = aux[ap]
    points = np.array(aux2)
    values = np.array(aux3)
    grid_z0 = griddata(points, values, (x_g, y_g), method='cubic')
    min_value = grid_z0[~np.isnan(grid_z0)].min()
    max_value = grid_z0[~np.isnan(grid_z0)].max()
    grid_z0 = (grid_z0 - min_value) / (max_value - min_value)
    return x_g, y_g, grid_z0


def plotInterpolatedAPImage(x_g: np.ndarray, y_g: np.ndarray, grid_z0: np.ndarray, ap: str):
    """
    Visualización de una AP interpolada para todas las posiciones (latitud, longitud) posibles

    Parameters:
    ----------
    x_g: array_like
       Array ordenado con las coordenadas de longitud de los RP
    y_g: array_like
       Array ordenado con las coordenadas de latitud de los RP
    grid_z0: array_like
       Array con los valores RSSI de la imagen interpolada del AP
    ap: str
       nombre del AP (Punto de acceso o wifi)

    Returns:
    ----------
    Visualization : none
       Visualización de la imagen interpolada del AP

    Examples
    --------
    Leer los datos del ground truth
    >>> import pandas as pd

    >>> df = pd.read_csv("../data/final_groundtruth/groundtruth.csv")

    Seleccionar las columnas del WiFi

    >>> wifi_train = df[df.columns[:8].tolist()]

    Interpolar la imagen del AP 480Invitados para todos los (latitud, longitud) posibles

    >>> x_g, y_g, grid_z0 = interpolateAPImage(wifi_train, "480Invitados")

    Visualizar la imagen interpolada

    >>> plotInterpolatedAPImage(x_g, y_g, grid_z0, "480Invitados")
    """

    plt.subplot(111)
    plt.scatter(x_g, y_g, s=2, marker='o', c=grid_z0, cmap=plt.cm.hsv)
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.title(f"Interpolated {ap} image")
    cbar = plt.colorbar()
    cbar.set_label('Chlor milligram m-3')
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.5, top=1.5, wspace=0.2, hspace=0.2)
    ax = plt.gca()
    ax.invert_xaxis()
    ax.set_facecolor('xkcd:black')
    plt.show()
