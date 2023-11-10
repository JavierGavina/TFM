from src.constants import constants
from src.dataloader import parse_windows
from src.dataloader import DataLoader
from src.imutils import plotAllAP, save_ap_gif
import numpy as np
import matplotlib.pyplot as plt
import os

root_dir = constants.outputs.PATH_OUTPUTS
rpmap = constants.outputs.rpmap.PATH_RPMAP
rpmap_300_overlapping = constants.outputs.rpmap.rpmap_300_overlapping
rpmap_300_sinOverlapping = constants.outputs.rpmap.rpmap_300_sinOverlapping
rpmap_28_overlapping = constants.outputs.rpmap.rpmap_28_overlapping

if __name__ == "__main__":
    print("creando directorios.........")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(rpmap, exist_ok=True)
    os.makedirs(rpmap_300_overlapping, exist_ok=True)
    os.makedirs(rpmap_300_sinOverlapping, exist_ok=True)
    os.makedirs(rpmap_28_overlapping, exist_ok=True)
    for x in ["gifs", "imagenes"]:
        os.makedirs(f"{rpmap_300_overlapping}/{x}", exist_ok=True)
        os.makedirs(f"{rpmap_300_sinOverlapping}/{x}", exist_ok=True)
        os.makedirs(f"{rpmap_28_overlapping}/{x}", exist_ok=True)

    print("cargando datos rpmap tamaño 300x300 sin overlapping.........")
    X_300_sin, y_300_sin, [x_coords_300_sin, y_coords_300_sin] = DataLoader(
        data_dir=f"{constants.data.FINAL_PATH}/groundtruth.csv",
        aps_list=constants.aps, batch_size=30, step_size=30,
        size_reference_point_map=300,
        return_axis_coords=True)()

    print("cargando datos rpmap tamaño 300x300 con overlapping.........")
    X_300_over, y_300_over, [x_coords_300_over, y_coords_300_over] = DataLoader(
        data_dir=f"{constants.data.FINAL_PATH}/groundtruth.csv",
        aps_list=constants.aps, batch_size=30, step_size=5,
        size_reference_point_map=300,
        return_axis_coords=True)()

    print("cargando datos rpmap tamaño 28x28 con overlapping.........")
    X_28_over, y_28_over, [x_coords_28_over, y_coords_28_over] = DataLoader(
        data_dir=f"{constants.data.FINAL_PATH}/groundtruth.csv",
        aps_list=constants.aps, batch_size=30, step_size=5,
        size_reference_point_map=28,
        return_axis_coords=True)()

    print("guardando imagenes rpmap 300x300 sin overlapping.........")
    plotAllAP(reference_point_map=X_300_sin[:, :, :, 0], labels=y_300_sin[:, 0], aps_list=constants.aps,
              path=f"{rpmap_300_sinOverlapping}/imagenes", save_ok=True, plot_ok=False)

    print("guardando imagenes rpmap 300x300 con overlapping.........")
    plotAllAP(reference_point_map=X_300_over[:, :, :, 0], labels=y_300_over[:, 0], aps_list=constants.aps,
              path=f"{rpmap_300_overlapping}/imagenes", save_ok=True, plot_ok=False)

    print("guardando imagenes rpmap 28x28 con overlapping.........")
    plotAllAP(reference_point_map=X_28_over[:, :, :, 0], labels=y_28_over[:, 0], aps_list=constants.aps,
              path=f"{rpmap_28_overlapping}/imagenes", save_ok=True, plot_ok=False)

    print("guardando gifs rpmap 300x300 sin overlapping.........")
    save_ap_gif(reference_point_map=X_300_sin[:, :, :, 0], x_g=x_coords_300_sin, y_g=y_coords_300_sin,
                aps_list=constants.aps,
                path=f"{rpmap_300_sinOverlapping}/gifs")

    print("guardando gifs rpmap 300x300 con overlapping.........")
    save_ap_gif(reference_point_map=X_300_over[:, :, :, 0], x_g=x_coords_300_over, y_g=y_coords_300_over,
                aps_list=constants.aps,
                path=f"{rpmap_300_overlapping}/gifs")

    print("guardando gifs rpmap 28x28 con overlapping.........")
    save_ap_gif(reference_point_map=X_28_over[:, :, :, 0], x_g=x_coords_28_over, y_g=y_coords_28_over,
                aps_list=constants.aps, reduced=True,
                path=f"{rpmap_28_overlapping}/gifs")
