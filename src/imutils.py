import os
import shutil
import tqdm

import IPython
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def displayGIF(path: str) -> IPython.display.Image:
    """
    Esta función se encarga de mostrar un GIF
    :param path: Dirección del GIF
    :return: Muestra el GIF
    """
    return IPython.display.Image(url=path)


def plotAllAP(reference_point_map: np.ndarray, labels: np.ndarray, aps_list: list, path: str = "reference_point_map",
              save_ok: bool = True, plot_ok: bool = True) -> None:
    """
    Esta función realiza un plot de todas las imágenes de los mapas de referencia continua para cada AP (wifi) en una misma figura.
    Opcionalmente, se puede guardar cada figura en un archivo PNG y mostrar la figura como salida.

    Parameters:
    -----------
    reference_point_map: np.ndarray
        Matriz con los valores interpolados del RSS de cada AP (wifi) para toda latitud y longitud en el espacio de muestreo

    labels: np.ndarray
        Etiquetas de los APs (wifi) para cada mapa de referencia continua

    aps_list: list
        Lista de APs (wifi) a considerar para la generación del mapa de referencia continua

    path: str = "reference_point_map"
        Ruta donde se guardarán las figuras de los mapas de referencia continua para cada AP (wifi)

    save_ok: bool = True
        Si es True, guarda las figuras de los mapas de referencia continua para cada AP (wifi) en un archivo PNG

    plot_ok: bool = True
        Si es True, muestra las figuras de los mapas de referencia continua para cada AP (wifi) como salida

    Returns:
    --------
    None:
        Esta función no devuelve ningún valor, solamente guarda las figuras de los mapas de referencia continua para cada AP (wifi) en un archivo PNG y/o muestra las figuras de los mapas de referencia continua para cada AP (wifi) como salida

    Examples:
    ---------
    >>> plotAllAP(reference_point_map=reference_point_map, labels=APLabel, aps_list=aps, path="reference_point_map", save_ok=True, plot_ok=True)
    """
    if save_ok:
        os.makedirs(path, exist_ok=True)  # Creamos la carpeta donde se guardarán las figuras
    nAPs = len(aps_list)  # Número de APs (wifi) a considerar
    n_samples = reference_point_map.shape[0]  # Número de muestras
    samples_per_AP = int(reference_point_map.shape[0] / nAPs)  # Número de muestras por AP (wifi)
    for id_ap in tqdm.tqdm(range(0, n_samples, samples_per_AP)):  # Iteramos sobre cada AP (wifi)
        ap = labels[id_ap]  # Etiqueta del AP (wifi)
        if samples_per_AP > 80:
            plt.figure(figsize=(20, 80), dpi=50)  # Creamos la figura
        else:
            plt.figure(figsize=(20, 15), dpi=50)  # Creamos la figura
        for idx in range(samples_per_AP):  # Iteramos sobre cada muestra
            plt.axis('off')
            plt.subplot(int(np.ceil(samples_per_AP / 7)), 7, idx + 1)  # Creamos un subplot para cada muestra
            plt.imshow(reference_point_map[idx + id_ap, :, :], cmap="seismic", vmin=0, vmax=1);
            plt.colorbar()  # Mostramos la imagen
            plt.gca().invert_xaxis()  # Invertimos eje x
            plt.gca().invert_yaxis()  # Invertimos eje y
            plt.title(f"{ap}, t={idx}")  # Configuramos el subplot
            plt.tight_layout()  # Ajustamos el subplot
        if save_ok:  # Si save_ok es True, guardamos la figura
            plt.savefig(f"{path}/{ap}.png")  # Guardamos la figura
        if plot_ok:  # Si plot_ok es True, mostramos la figura
            plt.show()  # Mostramos la figura
        plt.close()  # Cerramos la figura


def save_ap_gif(reference_point_map: np.ndarray, x_g: np.ndarray, y_g: np.ndarray, aps_list: list,
                reduced: bool = False, path="gifs"):
    """
    Esta función se encarga de guardar un GIF para cada AP (wifi) con los mapas de referencia continua para cada instante de tiempo.

    Parameters:
    -----------
    reference_point_map: np.ndarray
        Matriz con los valores interpolados del RSS de cada AP (wifi) para toda latitud y longitud en el espacio de muestreo

    x_g: np.ndarray
        Coordenadas longitud continuas del mapa de referencia continua

    y_g: np.ndarray
        Coordenadas latitud continuas del mapa de referencia continua

    aps_list: list
        Lista de APs (wifi) a considerar para la generación del mapa de referencia continua

    reduced: bool = False
        Si es True, se reduce el tamaño de la figura y se aumenta el tamaño de los puntos

    path: str = "gifs"
        Ruta donde se guardarán los GIFs de los mapas de referencia continua para cada AP (wifi)


    Returns:
    --------
        None:
            Esta función no devuelve ningún valor, solamente guarda los GIFs de los mapas de referencia continua para cada AP (wifi) en un archivo GIF

    Examples:
    ---------
    Carga de los datos:

    >>> from src import dataloader
    >>> X, y, [x_coords, y_coords] = dataloader.DataLoader(data_dir=f"../{constants.data.FINAL_PATH}/groundtruth.csv",
                                                           aps_list=constants.aps, batch_size=30, step_size=5,
                                                           size_reference_point_map=300,
                                                           return_axis_coords=False)()

    >>> RPMap, APLabel = X[:,:,:,0], y[:,0]
    >>> save_ap_gif(reference_point_map=RPMap, x_g=x_g, y_g=y_g, aps_list=aps, path="gifs")


    """
    os.makedirs(path, exist_ok=True)
    os.makedirs(f"{path}/temp", exist_ok=True)

    n_samples = reference_point_map.shape[0]
    n_APs = len(aps_list)
    samples_per_AP = int(n_samples / n_APs)

    if not reduced:
        figsize = (15, 15)
        fontsize = 20
        markersize = 2

    else:
        figsize = (5, 5)
        fontsize = 10
        markersize = 3

    for id_ap, id2_ap in tqdm.tqdm(enumerate(range(0, n_samples, samples_per_AP))):
        ap = aps_list[id_ap]
        os.makedirs(f"{path}/temp/{ap}", exist_ok=True)

        for idx in range(samples_per_AP):
            plt.figure(figsize=figsize)
            plt.scatter(x_g, y_g, s=markersize, marker='o', c=reference_point_map[idx + id2_ap, :, :].flatten(),
                        cmap="seismic", vmin=0,
                        vmax=1)
            plt.ylabel('Latitude', size=fontsize)
            plt.xlabel('Longitude', size=fontsize)
            plt.title(f"{ap}, t={idx}", fontsize=fontsize)
            cbar = plt.colorbar()
            cbar.set_label(f'RSS {ap}')
            ax = plt.gca()
            ax.invert_xaxis()
            ax.set_facecolor('xkcd:black')
            plt.savefig("{}/temp/{}/{}_t{:04d}.png".format(path, ap, ap, idx))
            plt.close()
        # conversión a gif
        folder_path = f"{path}/temp/{ap}"
        # Lista de nombres de archivo de imágenes PNG en la carpeta
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        # Ordenar los nombres de archivo en orden alfabético
        png_files.sort()
        # Lista de objetos Image para cada imagen PNG
        image_list = [Image.open(os.path.join(folder_path, f)) for f in png_files]
        # Guardar las imágenes como un archivo GIF animado
        image_list[0].save(f"{path}/{ap}.gif", save_all=True, append_images=image_list[1:], duration=350, loop=0)

    shutil.rmtree(f"{path}/temp")
