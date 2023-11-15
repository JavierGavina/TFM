import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

import sklearn as sk

sys.path.append("..")
from src.constants import constants
from src.dataloader import DataLoader, labelEncoding, labelDecoding
from positioning import utils

if __name__ == "__main__":
    # POSITIONING WITHOUT RPMAP
    groundtruth = pd.read_csv(f"../{constants.data.FINAL_PATH}/groundtruth.csv")
    X = groundtruth[constants.aps].to_numpy()
    y = groundtruth[["Longitude", "Latitude"]].to_numpy()

    # Splitting train and test
    Xtrain, Xtest, ytrain, ytest = sk.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

    # creating paths
    print("........ creating paths ........")
    os.makedirs(f"../{constants.outputs.positioning.positioning_path}", exist_ok=True)
    os.makedirs(f"../{constants.outputs.positioning.line_plot_metrics}", exist_ok=True)
    os.makedirs(f"../{constants.outputs.positioning.positioning_estimation}", exist_ok=True)
    os.makedirs(f"../{constants.outputs.positioning.preds_vs_true}", exist_ok=True)

    print("........ training hyperparameters knn ........")
    metricas_knn = utils.results_knn_training(Xtrain=Xtrain, Xtest=Xtest, ytrain=ytrain, ytest=ytest,
                                              k_neighbors=[x for x in range(1, 11)], sort_by_rmse=True)

    print(
        f"........ saving train results to ../{constants.outputs.positioning.positioning_estimation}/metricas_knn.csv........")
    metricas_knn.to_csv(f"../{constants.outputs.positioning.positioning_estimation}/metricas_knn.csv", index=False)

    print("........ training hyperparameters rf ........")
    metricas_rf = utils.results_rf_training(Xtrain=Xtrain, Xtest=Xtest, ytrain=ytrain, ytest=ytest,
                                            n_trees=[10, 100, 500, 1000, 2000, 5000], sort_by_rmse=True)
    print(
        f"........ saving train results to ../{constants.outputs.positioning.positioning_estimation}/metricas_rf.csv........"
    )
    metricas_rf.to_csv(f"../{constants.outputs.positioning.positioning_estimation}/metricas_rf.csv", index=False)

    print("........ saving line plot metrics knn ........")
    utils.knn_line_plot_metrics(results=metricas_knn, y_label="Error (m)", title="KNN",
                                path_out=f"../{constants.outputs.positioning.line_plot_metrics}",
                                save_ok=True)

    print("........ saving line plot metrics rf ........")
    utils.rf_line_plot_metrics(results=metricas_rf, y_label="Error (m)", title="RF",
                               path_out=f"../{constants.outputs.positioning.line_plot_metrics}",
                               save_ok=True)

    print("........ saving plot true vs pred ........")

    best_params_knn = int(metricas_knn.iloc[0]["k_neighbors"])
    best_params_rf = int(metricas_rf.iloc[0]["n_trees"])
    ypred_knn = sk.neighbors.KNeighborsRegressor(n_neighbors=best_params_knn).fit(Xtrain, ytrain).predict(Xtest)
    ypred_rf = sk.ensemble.RandomForestRegressor(n_estimators=best_params_rf, random_state=42).fit(Xtrain, ytrain).predict(Xtest)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(ytest[:, 0], ytest[:, 1], label="ytrue", s=100)
    plt.gca().invert_xaxis()
    plt.scatter(ypred_knn[:, 0], ypred_knn[:, 1], label="KNN", alpha=0.5)
    plt.gca().invert_xaxis(); plt.legend(); plt.title(f"KNN(k={best_params_knn})")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.subplot(1, 2, 2)
    plt.scatter(ytest[:, 0], ytest[:, 1], label="ytrue", s=100)
    plt.gca().invert_xaxis()
    plt.scatter(ypred_rf[:, 0], ypred_rf[:, 1], label="RF", alpha=0.5)
    plt.gca().invert_xaxis(); plt.legend(); plt.title(f"RF(n={best_params_rf})")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.savefig(f"../{constants.outputs.positioning.preds_vs_true}/preds_vs_true.png")
    plt.show()
