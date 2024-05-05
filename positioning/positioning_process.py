import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import seaborn as sns
import tqdm

sys.path.append("..")

from src.dataloader import DataLoader, get_radiomap_from_rpmap
from src.constants import constants
from src.preprocess import interpolacion_pixel_proximo
from models.gans_utils import DataAugmentation, get_path_cgan, get_path_wcgan_gp, get_path_wcgan, \
    incorpore_syntetic_data_to_real_data

from positioning.utils import metrics_dist_euclid_per_coord, get_metrics
import sklearn as sk
from multiprocessing import cpu_count


def plot_circles_around_points(ypred, ytrue, out_path):
    train_coords = np.array([(x, y) for x, y in constants.labels_dictionary_meters.values()])
    coords = [(x, y) for x, y in constants.labels_dictionary_meters_test.values()]
    coords_array = np.array(coords)
    dict_coords_to_colors = {coord: color for coord, color in zip(coords, constants.colors_test)}

    plt.figure(figsize=(20, 20), dpi=80)
    plt.scatter(coords_array[:, 0], coords_array[:, 1],
                color=[dict_coords_to_colors[(x[0], x[1])] for x in coords_array], s=200)

    # calcular el radio de cada punto de test con la máxima distancia euclidea
    radius = metrics_dist_euclid_per_coord(ytrue, ypred, option_return="max", return_all=False)
    for idx, (x, y) in enumerate(coords_array):
        if radius[idx] < 6:
            circle = Circle((x, y), radius[idx], color=dict_coords_to_colors[(x, y)],
                            fill=dict_coords_to_colors[(x, y)], alpha=0.4)
            plt.gca().add_patch(circle)

    # dibujar una cruz en cada punto de coords_array
    plt.scatter(coords_array[:, 0], coords_array[:, 1], color="black", s=200, marker="+")
    plt.gca().set_aspect('equal', adjustable='box')  # para que los ejes tengan la misma escala

    for pred, true in zip(ypred, ytrue):
        plt.scatter(pred[0], pred[1], color=dict_coords_to_colors[(true[0], true[1])], s=60, alpha=0.8)

    # rectangulo desde la esquina inferior izquierda hasta la superior derecha
    plt.gca().add_patch(
        plt.Rectangle((train_coords[:, 0].min() - 0.5, train_coords[:, 1].min() - 0.5), train_coords[:, 0].max() + 0.5,
                      train_coords[:, 1].max() + 0.5, fill=False, alpha=0.5, color="red", linewidth=2, linestyle="--"))

    filename = out_path.split("/")[-1].split(".")[0]
    plt.title(f"{filename} (only errors lower than 6 meters)", size=20)
    plt.xlabel("Longitude (meters)", size=20)
    plt.ylabel("Latitude (meters)", size=20)
    plt.gca().invert_xaxis()
    plt.savefig(out_path)
    plt.show()


def boxplot_metrics(metrics, path_out):
    data = metrics.copy()
    # Sorting by the mean
    mean_by_category = data.groupby('Model')['Mean_Euclidean_Dist'].mean().sort_values()
    df_sorted = data.set_index('Model').loc[mean_by_category.index].reset_index()

    # Setting up the style
    sns.set(style="whitegrid")

    # Creating a combined boxplot and jitter plot
    plt.figure(figsize=(10, 12))
    sns.boxplot(x='Model', y='Mean_Euclidean_Dist', data=df_sorted, order=mean_by_category.index)
    sns.stripplot(x='Model', y='Mean_Euclidean_Dist', data=df_sorted, order=mean_by_category.index, color='black',
                  jitter=True, alpha=0.7)

    # Showing the plot
    plt.title('Boxplot Metrics')
    plt.xlabel('Model')
    plt.ylabel('Mean Euclidean Distance on Test (meters) ')
    plt.xticks(size=10, rotation=45)
    plt.savefig(path_out)
    plt.show()


def obtain_data_gans_training(rpmap, cgan_generator, wcgan_generator, wcgan_gp_generator, x_coords, y_coords,
                              type="DA"):
    options = {
        "DA": 30,
        "SYN": 223
    }
    print("generating data............")
    cgan_rpmap, _ = cgan_generator(n_samples_per_label=options[type])
    wcgan_rpmap, _ = wcgan_generator(n_samples_per_label=options[type])
    wcgan_gp_rpmap, _ = wcgan_gp_generator(n_samples_per_label=options[type])

    if type == "DA":
        print("incorporating synthetic data to reals ............")
        cgan_rpmap = incorpore_syntetic_data_to_real_data(rpmap, cgan_rpmap)
        wcgan_rpmap = incorpore_syntetic_data_to_real_data(rpmap, wcgan_rpmap)
        wcgan_gp_rpmap = incorpore_syntetic_data_to_real_data(rpmap, wcgan_gp_rpmap)

    radiomap_cgan = get_radiomap_from_rpmap(cgan_rpmap, x_coords, y_coords)
    radiomap_wcgan = get_radiomap_from_rpmap(wcgan_rpmap, x_coords, y_coords)
    radiomap_wcgan_gp = get_radiomap_from_rpmap(wcgan_gp_rpmap, x_coords, y_coords)

    Xtrain_cgan, ytrain_cgan = radiomap_cgan[constants.aps].to_numpy(), radiomap_cgan[
        ["Longitude", "Latitude"]].to_numpy()
    Xtrain_wcgan, ytrain_wcgan = radiomap_wcgan[constants.aps].to_numpy(), radiomap_wcgan[
        ["Longitude", "Latitude"]].to_numpy()
    Xtrain_wcgan_gp, ytrain_wcgan_gp = radiomap_wcgan_gp[constants.aps].to_numpy(), radiomap_wcgan_gp[
        ["Longitude", "Latitude"]].to_numpy()

    return [Xtrain_cgan, ytrain_cgan], [Xtrain_wcgan, ytrain_wcgan], [Xtrain_wcgan_gp, ytrain_wcgan_gp]


def get_euclid_per_coord(ytrue, ypred):
    euclidean_distances = np.sqrt(np.sum((ypred - ytrue) ** 2, axis=1))
    coords_unique = np.array(list(constants.labels_dictionary_meters_test.values()))
    dist_max, dist_mean, dist_std = [], [], []
    for coord in coords_unique:
        indices = np.where(np.all(ytrue == coord, axis=1))[0]
        if indices.size > 0:
            start, end = indices[0], indices[-1] + 1
            dist_max.append(np.max(euclidean_distances[start:end]))
            dist_mean.append(np.mean(euclidean_distances[start:end]))
            dist_std.append(np.std(euclidean_distances[start:end]))
    return np.array(dist_max), np.array(dist_mean), np.array(dist_std)





if __name__ == "__main__":
    ORDERED_BY = "Kullback"
    positioning_metrics = "../outputs/positioning_metrics"
    ordered_by = f"{positioning_metrics}/orderedBy{ORDERED_BY}"
    radio_plots = f"{ordered_by}/radioplots"
    boxplots = f"{ordered_by}/boxplots"
    metrics = f"{ordered_by}/metrics"

    os.makedirs(positioning_metrics, exist_ok=True)
    os.makedirs(ordered_by, exist_ok=True)
    os.makedirs(radio_plots, exist_ok=True)
    os.makedirs(boxplots, exist_ok=True)
    os.makedirs(metrics, exist_ok=True)

    n_repeats = 10
    models = pd.DataFrame(columns=["Model", "n_repeat", "Mean_Euclidean_Dist", "RMSE", "MSE", "MAE"])

    print("=======================================================")
    print("LEYENDO DATOS SIN RPMAP")
    print("=======================================================")
    radiomap = pd.read_csv(f"../{constants.data.train.FINAL_PATH}/groundtruth.csv")
    radiomap = interpolacion_pixel_proximo(radiomap, threshold=30)
    Xtrain, ytrain = radiomap[constants.aps].to_numpy(), radiomap[["Longitude", "Latitude"]].to_numpy()

    print("=======================================================")
    print("LEYENDO DATOS TEST")
    print("=======================================================")
    radiomap_test = pd.read_csv(f"../{constants.data.test.FINAL_PATH}/groundtruth.csv")
    radiomap_test = interpolacion_pixel_proximo(radiomap_test, threshold=30)
    Xtest, ytest = radiomap_test[constants.aps].to_numpy(), radiomap_test[["Longitude", "Latitude"]].to_numpy()

    print("=======================================================")
    print("LEYENDO DATOS TRAIN CON RPMAP")
    print("=======================================================")

    get_loader = DataLoader(
        data_dir=f"../{constants.data.train.FINAL_PATH}/groundtruth.csv",
        aps_list=constants.aps, batch_size=30, step_size=5,
        size_reference_point_map=28, return_axis_coords=True
    )

    rpmap, rpmap_labels, [x_coords_tr, y_coords_tr] = get_loader()

    # Obtenemos radiomap
    radiomap_rpmap = get_radiomap_from_rpmap(rpmap, x_coords_tr, y_coords_tr)
    Xtrain_rpmap, ytrain_rpmap = radiomap_rpmap[constants.aps].to_numpy(), radiomap_rpmap[
        ["Longitude", "Latitude"]].to_numpy()

    # models of generators
    path_cgan = f"../{get_path_cgan(learning_rate=0.001, epoch=250)}"
    path_wcgan = f"../{get_path_wcgan(learning_rate=0.0005, n_critic=1, clip_value=0.1, epoch=250)}"
    path_wcgan_gp = f"../{get_path_wcgan_gp(learning_rate=0.0001, n_critic=5, gradient_penalty=15.0, epoch=250)}"

    cgan_generator = DataAugmentation(path_to_generator=path_cgan)
    wcgan_generator = DataAugmentation(path_to_generator=path_wcgan)
    wcgan_gp_generator = DataAugmentation(path_to_generator=path_wcgan_gp)

    print("=======================================================")
    print("ENTRENAMIENTO KNN Y RF PARA CADA MODELO")
    print("=======================================================")

    print("-----------PARAMETERS----------")
    print(">>  KNN(k_neighbors=15)")
    print(">>  RF(n_estimators=500)")

    print("=======================================================")
    print("TRAINING SIN RPMAP")
    print("=======================================================")
    for n_rep in tqdm.tqdm(range(n_repeats)):
        sin_rpmap_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count()-3).fit(Xtrain, ytrain)
        sin_rpmap_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count()-3).fit(Xtrain, ytrain)

        sin_rpmap_knn_preds = sin_rpmap_knn.predict(Xtest)
        sin_rpmap_rf_preds = sin_rpmap_rf.predict(Xtest)

        knn_rmse, knn_mae, knn_mse, knn_mean_eucl = get_metrics(sin_rpmap_knn_preds, ytest)
        rf_rmse, rf_mae, rf_mse, rf_mean_eucl = get_metrics(sin_rpmap_rf_preds, ytest)

        models.loc[len(models)] = ["SinRPMAP-KNN", n_rep, knn_mean_eucl, knn_rmse, knn_mse, knn_mae]
        models.loc[len(models)] = ["SinRPMAP-RF", n_rep, rf_mean_eucl, rf_rmse, rf_mse, rf_mae]

        if n_rep == 0:
            plot_circles_around_points(sin_rpmap_knn_preds, ytest, f"{radio_plots}/sin_rpmap_knn.png")
            plot_circles_around_points(sin_rpmap_rf_preds, ytest, f"{radio_plots}/sin_rpmap_rf.png")

    print("=======================================================")
    print("TRAINING CON RPMAP")
    print("=======================================================")

    for n_rep in tqdm.tqdm(range(n_repeats)):
        rpmap_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count()-3).fit(Xtrain_rpmap, ytrain_rpmap)
        rpmap_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count()-3).fit(Xtrain_rpmap, ytrain_rpmap)

        rpmap_knn_preds = rpmap_knn.predict(Xtest)
        rpmap_rf_preds = rpmap_rf.predict(Xtest)

        knn_rmse, knn_mae, knn_mse, knn_mean_eucl = get_metrics(rpmap_knn_preds, ytest)
        rf_rmse, rf_mae, rf_mse, rf_mean_eucl = get_metrics(rpmap_rf_preds, ytest)

        models.loc[len(models)] = ["RPMAP-KNN", n_rep, knn_mean_eucl, knn_rmse, knn_mse, knn_mae]
        models.loc[len(models)] = ["RPMAP-RF", n_rep, rf_mean_eucl, rf_rmse, rf_mse, rf_mae]

        if n_rep == 0:
            plot_circles_around_points(rpmap_knn_preds, ytest, f"{radio_plots}/rpmap_knn.png")
            plot_circles_around_points(rpmap_rf_preds, ytest, f"{radio_plots}/rpmap_rf.png")

    print("=======================================================")
    print("TRAINING CON RPMAP + GANS")
    print("=======================================================")

    for n_rep in tqdm.tqdm(range(n_repeats)):
        # Volvemos a generar los datos en cada iteración para comprobar la estabilidad de los gans
        generated_da = obtain_data_gans_training(rpmap=rpmap, cgan_generator=cgan_generator,
                                                 wcgan_generator=wcgan_generator, wcgan_gp_generator=wcgan_gp_generator,
                                                 x_coords=x_coords_tr, y_coords=y_coords_tr, type="DA")

        [Xtrain_cgan, ytrain_cgan], [Xtrain_wcgan, ytrain_wcgan], [Xtrain_wcgan_gp, ytrain_wcgan_gp] = generated_da

        # TRAINING CON GANS
        cgan_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count()-3).fit(Xtrain_cgan, ytrain_cgan)
        cgan_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count()-3).fit(Xtrain_cgan, ytrain_cgan)

        wcgan_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count()-3).fit(Xtrain_wcgan, ytrain_wcgan)
        wcgan_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count()-3).fit(Xtrain_wcgan, ytrain_wcgan)

        wcgan_gp_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count()-3).fit(Xtrain_wcgan_gp, ytrain_wcgan_gp)
        wcgan_gp_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count()-3).fit(Xtrain_wcgan_gp, ytrain_wcgan_gp)

        # PREDICCIONES CON GANS
        cgan_knn_preds = cgan_knn.predict(Xtest)
        cgan_rf_preds = cgan_rf.predict(Xtest)

        wcgan_knn_preds = wcgan_knn.predict(Xtest)
        wcgan_rf_preds = wcgan_rf.predict(Xtest)

        wcgan_gp_knn_preds = wcgan_gp_knn.predict(Xtest)
        wcgan_gp_rf_preds = wcgan_gp_rf.predict(Xtest)

        # METRICAS CON GANS
        cgan_knn_rmse, cgan_knn_mae, cgan_knn_mse, cgan_knn_mean_eucl = get_metrics(cgan_knn_preds, ytest)
        cgan_rf_rmse, cgan_rf_mae, cgan_rf_mse, cgan_rf_mean_eucl = get_metrics(cgan_rf_preds, ytest)

        wcgan_knn_rmse, wcgan_knn_mae, wcgan_knn_mse, wcgan_knn_mean_eucl = get_metrics(wcgan_knn_preds, ytest)
        wcgan_rf_rmse, wcgan_rf_mae, wcgan_rf_mse, wcgan_rf_mean_eucl = get_metrics(wcgan_rf_preds, ytest)

        wcgan_gp_knn_rmse, wcgan_gp_knn_mae, wcgan_gp_knn_mse, wcgan_gp_knn_mean_eucl = get_metrics(wcgan_gp_knn_preds,
                                                                                                    ytest)
        wcgan_gp_rf_rmse, wcgan_gp_rf_mae, wcgan_gp_rf_mse, wcgan_gp_rf_mean_eucl = get_metrics(wcgan_gp_rf_preds,
                                                                                                ytest)

        models.loc[len(models)] = ["cGAN (DA)-KNN", n_rep, cgan_knn_mean_eucl, cgan_knn_rmse, cgan_knn_mse,
                                   cgan_knn_mae]
        models.loc[len(models)] = ["cGAN (DA)-RF", n_rep, cgan_rf_mean_eucl, cgan_rf_rmse, cgan_rf_mse, cgan_rf_mae]

        models.loc[len(models)] = ["WcGAN (DA)-KNN", n_rep, wcgan_knn_mean_eucl, wcgan_knn_rmse, wcgan_knn_mse,
                                   wcgan_knn_mae]
        models.loc[len(models)] = ["WcGAN (DA)-RF", n_rep, wcgan_rf_mean_eucl, wcgan_rf_rmse, wcgan_rf_mse,
                                   wcgan_rf_mae]

        models.loc[len(models)] = ["WcGAN-GP (DA)-KNN", n_rep, wcgan_gp_knn_mean_eucl, wcgan_gp_knn_rmse,
                                   wcgan_gp_knn_mse,
                                   wcgan_gp_knn_mae]
        models.loc[len(models)] = ["WcGAN-GP (DA)-RF", n_rep, wcgan_gp_rf_mean_eucl, wcgan_gp_rf_rmse, wcgan_gp_rf_mse,
                                   wcgan_gp_rf_mae]

        if n_rep == 0:
            plot_circles_around_points(cgan_knn_preds, ytest, f"{radio_plots}/cgan_DA_knn.png")
            plot_circles_around_points(cgan_rf_preds, ytest, f"{radio_plots}/cgan_DA_rf.png")

            plot_circles_around_points(wcgan_knn_preds, ytest, f"{radio_plots}/wcgan_DA_knn.png")
            plot_circles_around_points(wcgan_rf_preds, ytest, f"{radio_plots}/wcgan_DA_rf.png")

            plot_circles_around_points(wcgan_gp_knn_preds, ytest, f"{radio_plots}/wcgan_gp_DA_knn.png")
            plot_circles_around_points(wcgan_gp_rf_preds, ytest, f"{radio_plots}/wcgan_gp_DA_rf.png")

    print("=======================================================")
    print("TRAINING CON RPMAP + GANS + SYN")
    print("=======================================================")

    for n_rep in tqdm.tqdm(range(n_repeats)):
        generated_syn = obtain_data_gans_training(rpmap=rpmap, cgan_generator=cgan_generator,
                                                  wcgan_generator=wcgan_generator,
                                                  wcgan_gp_generator=wcgan_gp_generator,
                                                  x_coords=x_coords_tr, y_coords=y_coords_tr, type="SYN")

        [Xtrain_cgan, ytrain_cgan], [Xtrain_wcgan, ytrain_wcgan], [Xtrain_wcgan_gp, ytrain_wcgan_gp] = generated_syn

        # TRAINING CON GANS
        cgan_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count()-3).fit(Xtrain_cgan, ytrain_cgan)
        cgan_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count()-3).fit(Xtrain_cgan, ytrain_cgan)

        wcgan_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count()-3).fit(Xtrain_wcgan, ytrain_wcgan)
        wcgan_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count()-3).fit(Xtrain_wcgan, ytrain_wcgan)

        wcgan_gp_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count()-3).fit(Xtrain_wcgan_gp, ytrain_wcgan_gp)
        wcgan_gp_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count()-3).fit(Xtrain_wcgan_gp, ytrain_wcgan_gp)

        # PREDICCIONES CON GANS
        cgan_knn_preds = cgan_knn.predict(Xtest)
        cgan_rf_preds = cgan_rf.predict(Xtest)

        wcgan_knn_preds = wcgan_knn.predict(Xtest)
        wcgan_rf_preds = wcgan_rf.predict(Xtest)

        wcgan_gp_knn_preds = wcgan_gp_knn.predict(Xtest)
        wcgan_gp_rf_preds = wcgan_gp_rf.predict(Xtest)

        # METRICAS CON GANS
        cgan_knn_rmse, cgan_knn_mae, cgan_knn_mse, cgan_knn_mean_eucl = get_metrics(cgan_knn_preds, ytest)
        cgan_rf_rmse, cgan_rf_mae, cgan_rf_mse, cgan_rf_mean_eucl = get_metrics(cgan_rf_preds, ytest)

        wcgan_knn_rmse, wcgan_knn_mae, wcgan_knn_mse, wcgan_knn_mean_eucl = get_metrics(wcgan_knn_preds, ytest)
        wcgan_rf_rmse, wcgan_rf_mae, wcgan_rf_mse, wcgan_rf_mean_eucl = get_metrics(wcgan_rf_preds, ytest)

        wcgan_gp_knn_rmse, wcgan_gp_knn_mae, wcgan_gp_knn_mse, wcgan_gp_knn_mean_eucl = get_metrics(wcgan_gp_knn_preds,
                                                                                                    ytest)
        wcgan_gp_rf_rmse, wcgan_gp_rf_mae, wcgan_gp_rf_mse, wcgan_gp_rf_mean_eucl = get_metrics(wcgan_gp_rf_preds,
                                                                                                ytest)

        models.loc[len(models)] = ["cGAN (SYN)-KNN", n_rep, cgan_knn_mean_eucl, cgan_knn_rmse, cgan_knn_mse,
                                      cgan_knn_mae]
        models.loc[len(models)] = ["cGAN (SYN)-RF", n_rep, cgan_rf_mean_eucl, cgan_rf_rmse, cgan_rf_mse, cgan_rf_mae]

        models.loc[len(models)] = ["WcGAN (SYN)-KNN", n_rep, wcgan_knn_mean_eucl, wcgan_knn_rmse, wcgan_knn_mse,
                                        wcgan_knn_mae]
        models.loc[len(models)] = ["WcGAN (SYN)-RF", n_rep, wcgan_rf_mean_eucl, wcgan_rf_rmse, wcgan_rf_mse,
                                        wcgan_rf_mae]
        models.loc[len(models)] = ["WcGAN-GP (SYN)-KNN", n_rep, wcgan_gp_knn_mean_eucl, wcgan_gp_knn_rmse,
                                        wcgan_gp_knn_mse,
                                        wcgan_gp_knn_mae]
        models.loc[len(models)] = ["WcGAN-GP (SYN)-RF", n_rep, wcgan_gp_rf_mean_eucl, wcgan_gp_rf_rmse, wcgan_gp_rf_mse,
                                        wcgan_gp_rf_mae]

        if n_rep == 0:
            plot_circles_around_points(cgan_knn_preds, ytest, f"{radio_plots}/cgan_SYN_knn.png")
            plot_circles_around_points(cgan_rf_preds, ytest, f"{radio_plots}/cgan_SYN_rf.png")

            plot_circles_around_points(wcgan_knn_preds, ytest, f"{radio_plots}/wcgan_SYN_knn.png")
            plot_circles_around_points(wcgan_rf_preds, ytest, f"{radio_plots}/wcgan_SYN_rf.png")

            plot_circles_around_points(wcgan_gp_knn_preds, ytest, f"{radio_plots}/wcgan_gp_SYN_knn.png")
            plot_circles_around_points(wcgan_gp_rf_preds, ytest, f"{radio_plots}/wcgan_gp_SYN_rf.png")

    print("=======================================================")
    print("GUARDANDO RESULTADOS")
    print("=======================================================")
    models.to_csv(f"{metrics}/metrics.csv", index=False)

    print("=======================================================")
    print("PLOTEANDO RESULTADOS")
    print("=======================================================")
    boxplot_metrics(models, f"{boxplots}/boxplot_metrics.png")

    print("=======================================================")
    print("TABLAS DE MEDIA +- STD")
    print("=======================================================")
    tabla_medias = models.groupby("Model").mean()[["Mean_Euclidean_Dist", "RMSE", "MSE", "MAE"]].reset_index()
    tabla_std = models.groupby("Model").std()[["Mean_Euclidean_Dist", "RMSE", "MSE", "MAE"]].reset_index()

    tabla_medias_std = tabla_medias.copy()
    for col in ["Mean_Euclidean_Dist", "RMSE", "MSE", "MAE"]:
        tabla_medias_std[col] = tabla_medias_std[col].round(3).astype(str) + " +- " + tabla_std[col].round(3).astype(
            str)
    # ordenar por la media de la distancia euclidea
    tabla_medias_std = tabla_medias_std.sort_values("Mean_Euclidean_Dist")
    tabla_medias_std.to_csv(f"{metrics}/tabla_resultados.csv", index=False)

