from src.dataloader import DataLoader, get_radiomap_from_rpmap
from src.constants import constants
from src.preprocess import interpolacion_pixel_proximo
from models.cGAN_300_300 import DataAugmentation
import tensorflow as tf

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

param_neighbors = [x for x in range(1, 15)]
param_n_estimators = [50, 65, 100, 200, 300, 500] + [x for x in range(1000, 5000, 500)]

if __name__ == "__main__":

    print("===============================")
    print("OBTENIENDO DATOS ORIGINALES")
    print("===============================")

    radiomap = pd.read_csv(f"{constants.data.train.FINAL_PATH}/groundtruth.csv")
    radiomap = interpolacion_pixel_proximo(radiomap, threshold=30)
    Xtrain, ytrain = radiomap[constants.aps].to_numpy(), radiomap[["Longitude", "Latitude"]].to_numpy()

    print("===============================")
    print("OBTENIENDO DATOS CON RADIOMAP AUMENTADO CON MALLADO CONTINUO")
    print("===============================")

    get_loader = DataLoader(data_dir=f"{constants.data.train.FINAL_PATH}/groundtruth.csv",
                            aps_list=constants.aps, batch_size=30, step_size=5,
                            size_reference_point_map=28, return_axis_coords=True)

    X, _, [x_coords, y_coords] = get_loader()
    rpmap = X[:, :, :, 0]
    radiomap_extended = get_radiomap_from_rpmap(rpmap, x_coords, y_coords)
    Xtrain_RPMAP, ytrain_RPMAP = radiomap_extended[constants.aps].to_numpy(), radiomap_extended[
        ["Longitude", "Latitude"]].to_numpy()

    print("===============================")
    print("OBTENIENDO DATOS AUMENTADOS CON GAN")
    print("===============================")

    data_augmentation = DataAugmentation(path_to_generator=f"{constants.outputs.models.cgan_28}/checkpoints/c_gan499.h5")
    generated, input_labels = data_augmentation(n_samples_per_label=30)

    samples_per_ap = int(rpmap.shape[0] / len(constants.aps))
    samples_generated_per_ap = int(generated.shape[0] / len(constants.aps))
    rpmap_ext = np.zeros((rpmap.shape[0] + samples_generated_per_ap * len(constants.aps), rpmap.shape[1], rpmap.shape[2]))
    count_gen = 0

    for n_ap, ap in enumerate(constants.aps):
        for batch_temporal in range(samples_per_ap):
            id_row = n_ap * samples_per_ap + batch_temporal
            rpmap_ext[count_gen, :, :] = rpmap[id_row, :, :]
            count_gen += 1
        for id_gen in range(samples_generated_per_ap):
            rpmap_ext[count_gen, :, :] = generated[id_gen + samples_generated_per_ap * n_ap]
            count_gen += 1

    radiomap_generated = get_radiomap_from_rpmap(rpmap_ext, x_coords, y_coords)
    Xtrain_generated, ytrain_generated = radiomap_generated[constants.aps].to_numpy(), radiomap_generated[["Longitude", "Latitude"]].to_numpy()

    print("===============================")
    print("OBTENIENDO DATOS SINTÉTICOS CON GAN")
    print("===============================")

    synthetic, _ = data_augmentation(n_samples_per_label=300)
    radiomap_synthetic = get_radiomap_from_rpmap(synthetic, x_coords, y_coords)
    Xtrain_synthetic, ytrain_synthetic = radiomap_synthetic[constants.aps].to_numpy(), radiomap_synthetic[["Longitude", "Latitude"]].to_numpy()

    print("===============================")
    print("OBTENIENDO DATOS DE TEST")
    print("===============================")

    radiomap_test = pd.read_csv(f"{constants.data.test.FINAL_PATH}/groundtruth.csv")
    radiomap_test = interpolacion_pixel_proximo(radiomap_test, threshold=30)
    Xtest, ytest = radiomap_test[constants.aps].to_numpy(), radiomap_test[["Longitude", "Latitude"]].to_numpy()

    print("===============================")
    print("ENTRENANDO MODELOS")
    print("===============================")

    print("===============================")
    print("ENTRENANDO CON LOS DATOS ORIGINALES")
    print("===============================")

    knn = GridSearchCV(KNeighborsRegressor(n_jobs=multiprocessing.cpu_count() - 3),
                       param_grid={"n_neighbors": param_neighbors},
                       cv=5, n_jobs=multiprocessing.cpu_count() - 3).fit(Xtrain, ytrain)

    rf = GridSearchCV(RandomForestRegressor(n_jobs=multiprocessing.cpu_count() - 3),
                      param_grid={"n_estimators": param_n_estimators},
                      cv=5, n_jobs=multiprocessing.cpu_count() - 3).fit(Xtrain, ytrain)

    print("best params: ")
    print(">>> knn: ", knn.best_params_)
    print(">>> rf: ", rf.best_params_)

    print("===============================")
    print("ENTRENANDO CON LOS DATOS DE MALLADO CONTINUO")
    print("===============================")

    knn_RPMAP = GridSearchCV(KNeighborsRegressor(n_jobs=multiprocessing.cpu_count() - 3),
                             param_grid={"n_neighbors": param_neighbors},
                             cv=5, n_jobs=multiprocessing.cpu_count() - 3).fit(Xtrain_RPMAP, ytrain_RPMAP)

    rf_RPMAP = GridSearchCV(RandomForestRegressor(n_jobs=multiprocessing.cpu_count() - 3),
                            param_grid={"n_estimators": param_n_estimators},
                            cv=5, n_jobs=multiprocessing.cpu_count() - 3).fit(Xtrain_RPMAP, ytrain_RPMAP)

    print("best params: ")
    print(">>> knn_RPMAP: ", knn_RPMAP.best_params_)
    print(">>> rf_RPMAP: ", rf_RPMAP.best_params_)

    print("===============================")
    print("ENTRENANDO CON LOS DATOS AUMENTADOS CON GAN")
    print("===============================")

    knn_generated = GridSearchCV(KNeighborsRegressor(n_jobs=multiprocessing.cpu_count() - 3),
                                 param_grid={"n_neighbors": param_neighbors},
                                 cv=5, n_jobs=multiprocessing.cpu_count() - 3).fit(Xtrain_generated, ytrain_generated)

    rf_generated = GridSearchCV(RandomForestRegressor(n_jobs=multiprocessing.cpu_count() - 3),
                                param_grid={"n_estimators": param_n_estimators},
                                cv=5, n_jobs=multiprocessing.cpu_count() - 3).fit(Xtrain_generated, ytrain_generated)

    print("best params: ")
    print(">>> knn_generated: ", knn_generated.best_params_)
    print(">>> rf_generated: ", rf_generated.best_params_)

    print("===============================")
    print("ENTRENANDO CON LOS DATOS SINTÉTICOS CON GAN")
    print("===============================")

    knn_synthetic = GridSearchCV(KNeighborsRegressor(n_jobs=multiprocessing.cpu_count() - 3),
                                 param_grid={"n_neighbors": param_neighbors},
                                 cv=5, n_jobs=multiprocessing.cpu_count() - 3).fit(Xtrain_synthetic, ytrain_synthetic)

    rf_synthetic = GridSearchCV(RandomForestRegressor(n_jobs=multiprocessing.cpu_count() - 3),
                                param_grid={"n_estimators": param_n_estimators},
                                cv=5, n_jobs=multiprocessing.cpu_count() - 3).fit(Xtrain_synthetic, ytrain_synthetic)

    print("best params: ")
    print(">>> knn_synthetic: ", knn_synthetic.best_params_)
    print(">>> rf_synthetic: ", rf_synthetic.best_params_)

    print("===============================")
    print("OBTENIENDO PREDICCIONES")
    print("===============================")

    ypred_knn = knn.predict(Xtest)
    ypred_rf = rf.predict(Xtest)

    ypred_knn_RPMAP = knn_RPMAP.predict(Xtest)
    ypred_rf_RPMAP = rf_RPMAP.predict(Xtest)

    ypred_knn_generated = knn_generated.predict(Xtest)
    ypred_rf_generated = rf_generated.predict(Xtest)

    ypred_knn_synthetic = knn_synthetic.predict(Xtest)
    ypred_rf_synthetic = rf_synthetic.predict(Xtest)

    print("===============================")
    print("OBTENIENDO DISTANCIAS EUCLIDEAS")
    print("===============================")

    euclidean_distances_knn = np.sqrt(np.sum((ypred_knn - ytest) ** 2, axis=1))
    euclidean_distances_rf = np.sqrt(np.sum((ypred_rf - ytest) ** 2, axis=1))

    euclidean_distances_knn_RPMAP = np.sqrt(np.sum((ypred_knn_RPMAP - ytest) ** 2, axis=1))
    euclidean_distances_rf_RPMAP = np.sqrt(np.sum((ypred_rf_RPMAP - ytest) ** 2, axis=1))

    euclidean_distances_knn_generated = np.sqrt(np.sum((ypred_knn_generated - ytest) ** 2, axis=1))
    euclidean_distances_rf_generated = np.sqrt(np.sum((ypred_rf_generated - ytest) ** 2, axis=1))

    euclidean_distances_knn_synthetic = np.sqrt(np.sum((ypred_knn_synthetic - ytest) ** 2, axis=1))
    euclidean_distances_rf_synthetic = np.sqrt(np.sum((ypred_rf_synthetic - ytest) ** 2, axis=1))

    print("===============================")
    print("OBTENIENDO DISTANCIAS EUCLIDEAS ORDENADAS")
    print("===============================")

    sorted_knn = np.sort(euclidean_distances_knn)
    sorted_rf = np.sort(euclidean_distances_rf)

    sorted_knn_RPMAP = np.sort(euclidean_distances_knn_RPMAP)
    sorted_rf_RPMAP = np.sort(euclidean_distances_rf_RPMAP)

    sorted_knn_generated = np.sort(euclidean_distances_knn_generated)
    sorted_rf_generated = np.sort(euclidean_distances_rf_generated)

    sorted_knn_synthetic = np.sort(euclidean_distances_knn_synthetic)
    sorted_rf_synthetic = np.sort(euclidean_distances_rf_synthetic)

    print("===============================")
    print("OBTENIENDO TABLA DE RESULTADOS")
    print("===============================")

    print(f">> KNN --- mean: {np.mean(sorted_knn):.2f}, 75% perc: {np.percentile(sorted_knn, 75):.2f}")
    print(f">> RF --- mean: {np.mean(sorted_rf):.2f}, 75% perc: {np.percentile(sorted_rf, 75):.2f}")
    print(
        f">> KNN RPMAP --- mean: {np.mean(sorted_knn_RPMAP):.2f}, 75% perc: {np.percentile(sorted_knn_RPMAP, 75):.2f}")
    print(f">> RF RPMAP --- mean: {np.mean(sorted_rf_RPMAP):.2f}, 75% perc: {np.percentile(sorted_rf_RPMAP, 75):.2f}")
    print(
        f">> KNN GAN --- mean: {np.mean(sorted_knn_generated):.2f}, 75% perc: {np.percentile(sorted_knn_generated, 75):.2f}")
    print(
        f">> RF GAN --- mean: {np.mean(sorted_rf_generated):.2f}, 75% perc: {np.percentile(sorted_rf_generated, 75):.2f}")
    print(
        f">> KNN SYNTHETIC --- mean: {np.mean(sorted_knn_synthetic):.2f}, 75% perc: {np.percentile(sorted_knn_synthetic, 75):.2f}")
    print(
        f">> RF SYNTHETIC --- mean: {np.mean(sorted_rf_synthetic):.2f}, 75% perc: {np.percentile(sorted_rf_synthetic, 75):.2f}")

    results = pd.DataFrame(columns=["method", "mean_error_distance", "percentile_75%_error"])
    results.loc[0] = [f"knn (k={knn.best_params_['n_neighbors']})", np.mean(sorted_knn), np.percentile(sorted_knn, 75)]
    results.loc[1] = [f"rf (n={rf.best_params_['n_estimators']})", np.mean(sorted_rf), np.percentile(sorted_rf, 75)]
    results.loc[2] = [f"knn RPMAP {knn_RPMAP.best_params_['n_neighbors']} ", np.mean(sorted_knn_RPMAP),
                      np.percentile(sorted_knn_RPMAP, 75)]
    results.loc[3] = [f"rf RPMAP (n={rf_RPMAP.best_params_['n_estimators']})", np.mean(sorted_rf_RPMAP),
                      np.percentile(sorted_rf_RPMAP, 75)]
    results.loc[4] = [f"knn GENERATED {knn_generated.best_params_['n_neighbors']}", np.mean(sorted_knn_generated),
                      np.percentile(sorted_knn_generated, 75)]
    results.loc[5] = [f"rf GENERATED (n={rf_generated.best_params_['n_estimators']})", np.mean(sorted_rf_generated),
                      np.percentile(sorted_rf_generated, 75)]
    results.loc[6] = [f"knn SYNTHETIC {knn_synthetic.best_params_['n_neighbors']}", np.mean(sorted_knn_synthetic),
                      np.percentile(sorted_knn_synthetic, 75)]
    results.loc[7] = [f"rf SYNTHETIC (n={rf_synthetic.best_params_['n_estimators']})", np.mean(sorted_rf_synthetic),
                      np.percentile(sorted_rf_synthetic, 75)]
    results.to_csv("tabla_resultados.csv", index=False)

    print("===============================")
    print("OBTENIENDO GRÁFICOS")
    print("===============================")

    plt.figure(figsize=(15, 7))
    plt.plot(sorted_knn, label="KNN")
    plt.plot(sorted_rf, label="RF")
    plt.plot(sorted_knn_RPMAP, linestyle="--", label="KNN RPMAP")
    plt.plot(sorted_rf_RPMAP, linestyle="--", label="RF RPMAP")
    plt.plot(sorted_knn_generated, linestyle="-.", label="KNN GAN")
    plt.plot(sorted_rf_generated, linestyle="-.", label="RF GAN")
    plt.plot(sorted_knn_synthetic, linestyle=":", label="KNN SYNTHETIC")
    plt.plot(sorted_rf_synthetic, linestyle=":", label="RF SYNTHETIC")
    plt.title("Distancias euclídeas")
    plt.legend()
    plt.savefig("resultados.png")
    plt.show()
