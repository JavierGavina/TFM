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

# diccionario modelo -> color
colors_models = {
    "SinRPMAP-KNN": "red",
    "SinRPMAP-RF": "red",
    "RPMAP-KNN": "green",
    "RPMAP-RF": "green",
    "cGAN (DA)-KNN": "orange",
    "cGAN (DA)-RF": "orange",
    "WcGAN (DA)-KNN": "purple",
    "WcGAN (DA)-RF": "purple",
    "WcGAN-GP (DA)-KNN": "blue",
    "WcGAN-GP (DA)-RF": "blue",
    "cGAN (SYN)-KNN": "purple",
    "cGAN (SYN)-RF": "purple",
    "WcGAN (SYN)-KNN": "yellow",
    "WcGAN (SYN)-RF": "yellow",
    "WcGAN-GP (SYN)-KNN": "black",
    "WcGAN-GP (SYN)-RF": "black"
}

dict_linestyles = {k: "--" if "KNN" in k else "-" for k, v in colors_models.items()}


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
    plt.title('Boxplot Metrics (with Coords)')
    plt.xlabel('Model')
    plt.ylabel('Mean Euclidean Distance on Test (meters) ')
    plt.xticks(size=10, rotation=45)
    plt.savefig(path_out)
    plt.show()


def obtain_data_gans_training(rpmap, cgan_generator, wcgan_generator, wcgan_gp_generator, x_coords, y_coords,
                              operation_type="DA"):
    options = {
        "DA": 30,
        "SYN": 223
    }
    print("generating data............")
    cgan_rpmap, _ = cgan_generator(n_samples_per_label=options[operation_type])
    wcgan_rpmap, _ = wcgan_generator(n_samples_per_label=options[operation_type])
    wcgan_gp_rpmap, _ = wcgan_gp_generator(n_samples_per_label=options[operation_type])

    if operation_type == "DA":
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


def update_metrics_per_coord(models, n_rep, ytrue, ypred, model_name):
    labels_dict_inverted = {v: k for k, v in constants.labels_dictionary_meters_test.items()}
    coords_unique = np.array(list(constants.labels_dictionary_meters_test.values()))
    for coord in coords_unique:
        indices = np.where(np.all(ytrue == coord, axis=1))[0]
        if indices.size > 0:
            mean_euclid_dist = np.mean(np.sqrt(np.sum((ypred[indices] - coord) ** 2, axis=1)))
            rmse = np.sqrt(np.mean((ypred[indices] - coord) ** 2))
            mae = np.mean(np.abs(ypred[indices] - coord))
            mse = np.mean((ypred[indices] - coord) ** 2)
            models.loc[len(models)] = [model_name, n_rep, labels_dict_inverted[(coord[0], coord[1])], mean_euclid_dist,
                                       rmse, mse, mae]


def plot_top_5_lineplots(metrics, path_out):
    plt.figure(figsize=(20, 10), dpi=80)

    # top 5 best models based on mean euclidean distance
    top_5 = metrics.groupby(["Model"]).mean().reset_index().sort_values(by="Mean_Euclidean_Dist").head(5)[
        "Model"].values

    for model in top_5:
        subset = metrics[metrics.Model == model]
        mean_subset = subset["Mean_Euclidean_Dist"].mean()
        perc_75 = np.percentile(subset["Mean_Euclidean_Dist"], 75)
        ordered = subset.groupby(["Model", "id_coord"]).mean().reset_index().sort_values(by="Mean_Euclidean_Dist").drop(
            ["Model", "n_repeat"], axis=1)

        plt.scatter([x for x in range(ordered.shape[0])], ordered["Mean_Euclidean_Dist"], color=colors_models[model])
        plt.plot([x for x in range(ordered.shape[0])], ordered["Mean_Euclidean_Dist"], linestyle=dict_linestyles[model],
                 color=colors_models[model], label=f"{model}, perc75 = {perc_75:.3f}, mean = {mean_subset:.3f}")

        plt.axhline(y=perc_75, color=colors_models[model], linestyle=dict_linestyles[model], alpha=0.8)
        # annotate points per id_coord
        for x, y, id_coord in zip([x for x in range(ordered.shape[0])], ordered["Mean_Euclidean_Dist"],
                                  ordered["id_coord"]):
            plt.annotate(id_coord, (x + 0.1, y - 0.1), fontsize=15)

    plt.legend()
    plt.title("Mean Euclidean Distance on TOP 5 Models, per Reference Point on Test", size=20)
    plt.xlabel("Sorted Reference Point", size=15)
    plt.ylabel("Mean Euclidean Distance (on meters)", size=15)
    plt.savefig(path_out)
    plt.show()


if __name__ == "__main__":
    ORDERED_BY = "Kullback"
    positioning_metrics = "../outputs/positioning_metrics_per_coord"
    ordered_by = f"{positioning_metrics}/orderedBy{ORDERED_BY}"
    radio_plots = f"{ordered_by}/radioplots"
    boxplots = f"{ordered_by}/boxplots"
    metrics = f"{ordered_by}/metrics"
    lineplot = f"{ordered_by}/lineplot"

    os.makedirs(positioning_metrics, exist_ok=True)
    os.makedirs(ordered_by, exist_ok=True)
    os.makedirs(radio_plots, exist_ok=True)
    os.makedirs(boxplots, exist_ok=True)
    os.makedirs(metrics, exist_ok=True)
    os.makedirs(lineplot, exist_ok=True)
    #
    # n_repeats = 10
    # models = pd.DataFrame(columns=["Model", "n_repeat", "id_coord", "Mean_Euclidean_Dist", "RMSE", "MSE", "MAE"])
    #
    # print("=======================================================")
    # print("LEYENDO DATOS SIN RPMAP")
    # print("=======================================================")
    # radiomap = pd.read_csv(f"../{constants.data.train.FINAL_PATH}/groundtruth.csv")
    # radiomap = interpolacion_pixel_proximo(radiomap, threshold=30)
    # Xtrain, ytrain = radiomap[constants.aps].to_numpy(), radiomap[["Longitude", "Latitude"]].to_numpy()
    #
    # print("=======================================================")
    # print("LEYENDO DATOS TEST")
    # print("=======================================================")
    # radiomap_test = pd.read_csv(f"../{constants.data.test.FINAL_PATH}/groundtruth.csv")
    # radiomap_test = interpolacion_pixel_proximo(radiomap_test, threshold=30)
    # Xtest, ytest = radiomap_test[constants.aps].to_numpy(), radiomap_test[["Longitude", "Latitude"]].to_numpy()
    #
    # print("=======================================================")
    # print("LEYENDO DATOS TRAIN CON RPMAP")
    # print("=======================================================")
    #
    # get_loader = DataLoader(
    #     data_dir=f"../{constants.data.train.FINAL_PATH}/groundtruth.csv",
    #     aps_list=constants.aps, batch_size=30, step_size=5,
    #     size_reference_point_map=28, return_axis_coords=True
    # )
    #
    # rpmap, rpmap_labels, [x_coords_tr, y_coords_tr] = get_loader()
    #
    # # Obtenemos radiomap
    # radiomap_rpmap = get_radiomap_from_rpmap(rpmap, x_coords_tr, y_coords_tr)
    # Xtrain_rpmap, ytrain_rpmap = radiomap_rpmap[constants.aps].to_numpy(), radiomap_rpmap[
    #     ["Longitude", "Latitude"]].to_numpy()
    #
    # # models of generators
    # path_cgan = f"../{get_path_cgan(learning_rate=0.001, epoch=250)}"
    # path_wcgan = f"../{get_path_wcgan(learning_rate=0.0005, n_critic=1, clip_value=0.1, epoch=250)}"
    # path_wcgan_gp = f"../{get_path_wcgan_gp(learning_rate=0.0001, n_critic=5, gradient_penalty=15.0, epoch=250)}"
    #
    # cgan_generator = DataAugmentation(path_to_generator=path_cgan)
    # wcgan_generator = DataAugmentation(path_to_generator=path_wcgan)
    # wcgan_gp_generator = DataAugmentation(path_to_generator=path_wcgan_gp)
    #
    # print("=======================================================")
    # print("ENTRENAMIENTO KNN Y RF PARA CADA MODELO")
    # print("=======================================================")
    #
    # print("-----------PARAMETERS----------")
    # print(">>  KNN(k_neighbors=15)")
    # print(">>  RF(n_estimators=500)")
    #
    # print("=======================================================")
    # print("TRAINING SIN RPMAP")
    # print("=======================================================")
    # for n_rep in tqdm.tqdm(range(n_repeats)):
    #     sin_rpmap_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count() - 3).fit(Xtrain, ytrain)
    #     sin_rpmap_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count() - 3).fit(Xtrain, ytrain)
    #
    #     sin_rpmap_knn_preds = sin_rpmap_knn.predict(Xtest)
    #     sin_rpmap_rf_preds = sin_rpmap_rf.predict(Xtest)
    #
    #     update_metrics_per_coord(models, n_rep, ytest, sin_rpmap_knn_preds, "SinRPMAP-KNN")
    #     update_metrics_per_coord(models, n_rep, ytest, sin_rpmap_rf_preds, "SinRPMAP-RF")
    #
    #     if n_rep == 0:
    #         plot_circles_around_points(sin_rpmap_knn_preds, ytest, f"{radio_plots}/sin_rpmap_knn.png")
    #         plot_circles_around_points(sin_rpmap_rf_preds, ytest, f"{radio_plots}/sin_rpmap_rf.png")
    #
    # print("=======================================================")
    # print("TRAINING CON RPMAP")
    # print("=======================================================")
    #
    # for n_rep in tqdm.tqdm(range(n_repeats)):
    #     rpmap_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count() - 3).fit(Xtrain_rpmap,
    #                                                                                              ytrain_rpmap)
    #     rpmap_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count() - 3).fit(Xtrain_rpmap,
    #                                                                                                ytrain_rpmap)
    #
    #     rpmap_knn_preds = rpmap_knn.predict(Xtest)
    #     rpmap_rf_preds = rpmap_rf.predict(Xtest)
    #
    #     update_metrics_per_coord(models, n_rep, ytest, rpmap_knn_preds, "RPMAP-KNN")
    #     update_metrics_per_coord(models, n_rep, ytest, rpmap_rf_preds, "RPMAP-RF")
    #
    #     if n_rep == 0:
    #         plot_circles_around_points(rpmap_knn_preds, ytest, f"{radio_plots}/rpmap_knn.png")
    #         plot_circles_around_points(rpmap_rf_preds, ytest, f"{radio_plots}/rpmap_rf.png")
    #
    # print("=======================================================")
    # print("TRAINING CON RPMAP + GANS")
    # print("=======================================================")
    #
    # for n_rep in tqdm.tqdm(range(n_repeats)):
    #     # Volvemos a generar los datos en cada iteración para comprobar la estabilidad de los gans
    #     generated_da = obtain_data_gans_training(rpmap=rpmap, cgan_generator=cgan_generator,
    #                                              wcgan_generator=wcgan_generator, wcgan_gp_generator=wcgan_gp_generator,
    #                                              x_coords=x_coords_tr, y_coords=y_coords_tr, operation_type="DA")
    #
    #     [Xtrain_cgan, ytrain_cgan], [Xtrain_wcgan, ytrain_wcgan], [Xtrain_wcgan_gp, ytrain_wcgan_gp] = generated_da
    #
    #     # TRAINING CON GANS
    #     cgan_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count() - 3).fit(Xtrain_cgan,
    #                                                                                             ytrain_cgan)
    #     cgan_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count() - 3).fit(Xtrain_cgan,
    #                                                                                               ytrain_cgan)
    #
    #     wcgan_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count() - 3).fit(Xtrain_wcgan,
    #                                                                                              ytrain_wcgan)
    #     wcgan_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count() - 3).fit(Xtrain_wcgan,
    #                                                                                                ytrain_wcgan)
    #
    #     wcgan_gp_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count() - 3).fit(Xtrain_wcgan_gp,
    #                                                                                                 ytrain_wcgan_gp)
    #     wcgan_gp_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count() - 3).fit(Xtrain_wcgan_gp,
    #                                                                                                   ytrain_wcgan_gp)
    #
    #     # PREDICCIONES CON GANS
    #     cgan_knn_preds = cgan_knn.predict(Xtest)
    #     cgan_rf_preds = cgan_rf.predict(Xtest)
    #
    #     wcgan_knn_preds = wcgan_knn.predict(Xtest)
    #     wcgan_rf_preds = wcgan_rf.predict(Xtest)
    #
    #     wcgan_gp_knn_preds = wcgan_gp_knn.predict(Xtest)
    #     wcgan_gp_rf_preds = wcgan_gp_rf.predict(Xtest)
    #
    #     # METRICAS CON GANS
    #     update_metrics_per_coord(models, n_rep, ytest, cgan_knn_preds, "cGAN (DA)-KNN")
    #     update_metrics_per_coord(models, n_rep, ytest, cgan_rf_preds, "cGAN (DA)-RF")
    #
    #     update_metrics_per_coord(models, n_rep, ytest, wcgan_knn_preds, "WcGAN (DA)-KNN")
    #     update_metrics_per_coord(models, n_rep, ytest, wcgan_rf_preds, "WcGAN (DA)-RF")
    #
    #     update_metrics_per_coord(models, n_rep, ytest, wcgan_gp_knn_preds, "WcGAN-GP (DA)-KNN")
    #     update_metrics_per_coord(models, n_rep, ytest, wcgan_gp_rf_preds, "WcGAN-GP (DA)-RF")
    #
    #     if n_rep == 0:
    #         plot_circles_around_points(cgan_knn_preds, ytest, f"{radio_plots}/cgan_DA_knn.png")
    #         plot_circles_around_points(cgan_rf_preds, ytest, f"{radio_plots}/cgan_DA_rf.png")
    #
    #         plot_circles_around_points(wcgan_knn_preds, ytest, f"{radio_plots}/wcgan_DA_knn.png")
    #         plot_circles_around_points(wcgan_rf_preds, ytest, f"{radio_plots}/wcgan_DA_rf.png")
    #
    #         plot_circles_around_points(wcgan_gp_knn_preds, ytest, f"{radio_plots}/wcgan_gp_DA_knn.png")
    #         plot_circles_around_points(wcgan_gp_rf_preds, ytest, f"{radio_plots}/wcgan_gp_DA_rf.png")
    #
    # print("=======================================================")
    # print("TRAINING CON RPMAP + GANS + SYN")
    # print("=======================================================")
    #
    # for n_rep in tqdm.tqdm(range(n_repeats)):
    #     generated_syn = obtain_data_gans_training(rpmap=rpmap, cgan_generator=cgan_generator,
    #                                               wcgan_generator=wcgan_generator,
    #                                               wcgan_gp_generator=wcgan_gp_generator,
    #                                               x_coords=x_coords_tr, y_coords=y_coords_tr, operation_type="SYN")
    #
    #     [Xtrain_cgan, ytrain_cgan], [Xtrain_wcgan, ytrain_wcgan], [Xtrain_wcgan_gp, ytrain_wcgan_gp] = generated_syn
    #
    #     # TRAINING CON GANS
    #     cgan_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count() - 3).fit(Xtrain_cgan,
    #                                                                                             ytrain_cgan)
    #     cgan_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count() - 3).fit(Xtrain_cgan,
    #                                                                                               ytrain_cgan)
    #
    #     wcgan_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count() - 3).fit(Xtrain_wcgan,
    #                                                                                              ytrain_wcgan)
    #     wcgan_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count() - 3).fit(Xtrain_wcgan,
    #                                                                                                ytrain_wcgan)
    #
    #     wcgan_gp_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=15, n_jobs=cpu_count() - 3).fit(Xtrain_wcgan_gp,
    #                                                                                                 ytrain_wcgan_gp)
    #     wcgan_gp_rf = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count() - 3).fit(Xtrain_wcgan_gp,
    #                                                                                                   ytrain_wcgan_gp)
    #
    #     # PREDICCIONES CON GANS
    #     cgan_knn_preds = cgan_knn.predict(Xtest)
    #     cgan_rf_preds = cgan_rf.predict(Xtest)
    #
    #     wcgan_knn_preds = wcgan_knn.predict(Xtest)
    #     wcgan_rf_preds = wcgan_rf.predict(Xtest)
    #
    #     wcgan_gp_knn_preds = wcgan_gp_knn.predict(Xtest)
    #     wcgan_gp_rf_preds = wcgan_gp_rf.predict(Xtest)
    #
    #     # METRICAS CON GANS
    #     update_metrics_per_coord(models, n_rep, ytest, cgan_knn_preds, "cGAN (SYN)-KNN")
    #     update_metrics_per_coord(models, n_rep, ytest, cgan_rf_preds, "cGAN (SYN)-RF")
    #
    #     update_metrics_per_coord(models, n_rep, ytest, wcgan_knn_preds, "WcGAN (SYN)-KNN")
    #     update_metrics_per_coord(models, n_rep, ytest, wcgan_rf_preds, "WcGAN (SYN)-RF")
    #
    #     update_metrics_per_coord(models, n_rep, ytest, wcgan_gp_knn_preds, "WcGAN-GP (SYN)-KNN")
    #     update_metrics_per_coord(models, n_rep, ytest, wcgan_gp_rf_preds, "WcGAN-GP (SYN)-RF")
    #
    #     if n_rep == 0:
    #         plot_circles_around_points(cgan_knn_preds, ytest, f"{radio_plots}/cgan_SYN_knn.png")
    #         plot_circles_around_points(cgan_rf_preds, ytest, f"{radio_plots}/cgan_SYN_rf.png")
    #
    #         plot_circles_around_points(wcgan_knn_preds, ytest, f"{radio_plots}/wcgan_SYN_knn.png")
    #         plot_circles_around_points(wcgan_rf_preds, ytest, f"{radio_plots}/wcgan_SYN_rf.png")
    #
    #         plot_circles_around_points(wcgan_gp_knn_preds, ytest, f"{radio_plots}/wcgan_gp_SYN_knn.png")
    #         plot_circles_around_points(wcgan_gp_rf_preds, ytest, f"{radio_plots}/wcgan_gp_SYN_rf.png")
    #
    # print("=======================================================")
    # print("GUARDANDO RESULTADOS")
    # print("=======================================================")
    # models.to_csv(f"{metrics}/metrics.csv", index=False)
    #
    # print("=======================================================")
    # print("PLOTEANDO RESULTADOS")
    # print("=======================================================")
    # boxplot_metrics(models, f"{boxplots}/boxplot_metrics.png")
    #
    models = pd.read_csv(f"{metrics}/metrics.csv")

    print("=======================================================")
    print("PLOTEANDO TOP 5")
    print("=======================================================")

    plot_top_5_lineplots(models, f"{lineplot}/lineplot.png")

    print("=======================================================")
    print("TABLAS DE MEDIA +- STD")
    print("=======================================================")

    tabla_medias = models.groupby(["Model"]).mean()[
        ["Mean_Euclidean_Dist", "RMSE", "MSE", "MAE"]].reset_index()
    tabla_std = models.groupby(["Model"]).std()[["Mean_Euclidean_Dist", "RMSE", "MSE", "MAE"]].reset_index()
    print(tabla_medias)
    tabla_medias_std = tabla_medias.copy()
    for col in ["Mean_Euclidean_Dist", "RMSE", "MSE", "MAE"]:
        tabla_medias_std[col] = tabla_medias_std[col].round(3).astype(str) + " +- " + tabla_std[col].round(3).astype(
            str)
    # ordenar por la media de la distancia euclidea
    tabla_medias_std = tabla_medias_std.sort_values("Mean_Euclidean_Dist")
    tabla_medias_std.to_csv(f"{metrics}/tabla_resultados.csv", index=False)
