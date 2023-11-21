import pandas as pd
import numpy as np

import os
import glob
import sys

from tqdm import tqdm

sys.path.append("..")
from src.constants import constants


def correctWifiFP(wifi_data: pd.DataFrame, t_max_sampling: int, dict_labels_to_meters: dict) -> pd.DataFrame:
    """
    Coge el fingerprint de los datos y aplica el siguiente preprocesado:
            1) Ajuste de frecuencia muestral a 1 muestra / segundo
                    - Cálculo de la media agrupada por segundo y por label en cada una de las balizas
            2) Ajuste de datos ausentes:
                    - Los segundos sin recogida de datos son reemplazados por el mínimo RSS del dataset menos 1
            3) Se escalan los valores entre 0 y 1 (siendo 0 la representación del dato ausente)
                    X = (X - X_min + 1)/X_max

    Parameters:
    -----------
    wifi_data: pd.DataFrame
        pd.DataFrame de los datos correspondientes al WiFi

    t_max_sampling: int
        Tiempo máximo de muestreo por label

    dict_labels_to_meters: dict
        Diccionario que transforma el label a (longitud, latitud) en metros

    Returns:
    --------
    wifi_corrected: pd.DataFrame
        pd.DataFrame de los datos corregidos

    Example:
    --------
    >>> t_max_sampling = constants.T_MAX_SAMPLING # 1140 segundos de recogida de datos por cada label en train (60s en test)
    >>> dict_labels_to_meters = constants.labels_dictionary_meters
    >>> wifi_corrected = correctWifiFP(wifi_data, t_max_sampling=t_max_sampling, dict_labels_to_meters=dict_labels_to_meters)
    """

    # Cogemos los datos de las balizas que han aparecido en más de una recogida de datos
    aux = wifi_data.query(f'Name_SSID in {constants.aps}')

    # Creamos un dataframe con todos los intervalos de tiempo y todas las balizas
    labels, intervalos_tiempo, ssids = [], [], []
    for lab in range(len(dict_labels_to_meters)):
        for ts in range(0, t_max_sampling + 1, 1):
            for ssid in constants.aps:
                labels.append(lab)
                intervalos_tiempo.append(ts)
                ssids.append(ssid)
    df_intervalos = pd.DataFrame(
        {'Label': labels, 'AppTimestamp(s)': intervalos_tiempo, 'Name_SSID': ssids})  # Dataframe de intervalos

    aux["AppTimestamp(s)"] = aux["AppTimestamp(s)"].round()  # Redondeamos el timestamp a 0 decimales
    aux = aux.groupby(["Label", "AppTimestamp(s)", "Name_SSID"]).mean()[
        "RSS"].reset_index()  # Agrupamos por label, timestamp y ssid
    aux_corrected = pd.merge(df_intervalos, aux, on=["Label", "AppTimestamp(s)", "Name_SSID"],
                             how="left")  # Unimos con el dataframe de intervalos
    # Reemplazamos los valores ausentes por el mínimo global
    aux_corrected = aux_corrected.pivot(index=['Label', 'AppTimestamp(s)'], columns='Name_SSID')[
        "RSS"].reset_index()  # Pivotamos el dataframe

    # Reescalado de valores entre 0 y 1
    aux_corrected[constants.aps] -= aux_corrected[constants.aps].min().min() - 1
    aux_corrected[constants.aps] = aux_corrected[constants.aps].fillna(0)
    aux_corrected[constants.aps] /= aux_corrected[constants.aps].max().max()
    aux_corrected[["Longitude", "Latitude"]] = [dict_labels_to_meters[x] for x in
                                                aux_corrected.Label]  # Añadimos la longitud y latitud
    orden_wifi_columnas = ["AppTimestamp(s)"] + \
                          [x for x in aux_corrected.columns if
                           x not in ["AppTimestamp(s)", "Latitude", "Longitude", "Label"]] + \
                          ["Latitude", "Longitude", "Label"]
    wifi_corrected = aux_corrected[orden_wifi_columnas]  # Ordenamos las columnas
    return wifi_corrected


def correctMetrics(data: pd.DataFrame, columns_to_correct: list, t_max_sampling: int,
                   dict_labels_to_meters: dict) -> pd.DataFrame:
    """
    Coge un dataset de métricas y aplica el siguiente preprocesado:
            1) Ajuste de frecuencia muestral a 1 muestra / segundo
                    - Cálculo de la media agrupada por segundo y por label en cada métrica

            2) Se escalan los valores entre 0 y 1
                    X = (X + |X_min|)/X_max

    Parameters:
    -----------
    data: pd.DataFrame
        pd.DataFrame de los datos correspondientes a las métricas
    columns_to_correct: list
        Lista de columnas a corregir
    t_max_sampling: int
        Tiempo máximo de muestreo por label
    dict_labels_to_meters: dict
        Diccionario que transforma el label a (longitud, latitud) en metros

    Returns:
    --------
    data_corrected: pd.DataFrame
        pd.DataFrame de los datos corregidos

    Example:
    --------
    Corregir giroscopio:

    >>> gyroscope_corrected = correctMetrics(gyroscope, ["Gyr_X", "Gyr_Y", "Gyr_Z"])

    Corregir acelerómetro:

    >>> accelerometer_corrected = correctMetrics(accelerometer, ["Acc_X", "Acc_Y", "Acc_Z"])

    Corregir magnetómetro:

    >>> magnetometer_corrected = correctMetrics(magnetometer, ["Mag_X", "Mag_Y", "Mag_Z"])
    """

    intervalos_tiempo = np.arange(0, t_max_sampling + 1, 1)
    df_intervalos = pd.DataFrame({'AppTimestamp(s)': intervalos_tiempo})
    data_corrected = pd.DataFrame(
        columns=["AppTimestamp(s)"] + columns_to_correct +
                ["Latitude", "Longitude", "Label"])
    for label, position in dict_labels_to_meters.items():
        aux = data.query(f"Label=={label}")
        aux["AppTimestamp(s)"] = aux["AppTimestamp(s)"].round()
        aux = aux.groupby("AppTimestamp(s)").mean()[
            [columns_to_correct[0], columns_to_correct[1], columns_to_correct[2]]].reset_index()
        aux_corrected = pd.merge(df_intervalos, aux, on="AppTimestamp(s)", how="left").set_index("AppTimestamp(s)")

        # Interpolación datos ausentes
        aux_corrected[columns_to_correct[0]] = aux_corrected[columns_to_correct[0]].interpolate(method="cubic")
        aux_corrected[columns_to_correct[1]] = aux_corrected[columns_to_correct[1]].interpolate(method="cubic")
        aux_corrected[columns_to_correct[2]] = aux_corrected[columns_to_correct[2]].interpolate(method="cubic")

        minimo_global = np.abs(aux_corrected[columns_to_correct].min().min())
        aux_corrected[columns_to_correct] = aux_corrected[columns_to_correct] + minimo_global
        maximo_global = aux_corrected[columns_to_correct].max().max()
        aux_corrected[columns_to_correct] = aux_corrected[columns_to_correct] / maximo_global
        aux_corrected[["Latitude", "Longitude", "Label"]] = [position[1], position[0], label]

        data_corrected = pd.concat([data_corrected, aux_corrected.reset_index()], ignore_index=False)
    return data_corrected


def interpolacionConcatenada(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica una interpolación a los datos de WiFi, de forma que si hay un hueco en el timestamp, se rellena con el valor anterior si y solo si el valor anterior y posterior son iguales

    Parameters:
    -----------
    data: pd.DataFrame
        pd.DataFrame de los datos correspondientes al WiFi

    Returns:
    --------
    new_data: pd.DataFrame
        pd.DataFrame de los datos corregidos

    Example:
    --------
    >>> wifi_corrected = interpolacionConcatenada(wifi_data)
    """
    new_data = data.copy()
    others = ["AppTimestamp(s)", "Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z", "Mag_X", "Mag_Y", "Mag_Z",
              "Latitude", "Longitude", "Label"]
    columnas_wifi = [x for x in new_data.columns if x not in others]
    idx_wifi = [idx for idx, c in enumerate(new_data.columns) if c in columnas_wifi]  # Indices de las columnas de wifi
    n_timestamp = new_data.query("Label==0")["AppTimestamp(s)"].max()  # Maximo timestamp de cada label

    # Recorremos cada columna de wifi
    for wifi in idx_wifi:
        # Recorremos cada segundo
        for idx_row in range(new_data.shape[0] - 1):
            # Si no es el primer índice, no es multiplo del maximo timestamp y el valor anterior y posterior son iguales
            if idx_row > 0 and idx_row % n_timestamp != 0 and idx_row % (n_timestamp + 1) != 0 \
                    and new_data.iloc[idx_row - 1, wifi] > 0 \
                    and new_data.iloc[idx_row + 1, wifi] > 0 \
                    and new_data.iloc[idx_row, wifi] == 0 \
                    and new_data.iloc[idx_row - 1, wifi] == new_data.iloc[idx_row + 1, wifi]:
                new_data.iloc[idx_row, wifi] = new_data.iloc[idx_row - 1, wifi]  # Rellenamos con el valor anterior

            # Si es el primer índice y el valor posterior es mayor que 0 y el valor actual es 0
            if idx_row == 0 or idx_row % (n_timestamp + 1) == 0 \
                    and new_data.iloc[idx_row + 1, wifi] > 0 \
                    and new_data.iloc[idx_row, wifi] == 0:
                new_data.iloc[idx_row, wifi] = new_data.iloc[idx_row + 1, wifi]  # Rellenamos con el valor posterior

            # Si es múltiplo del maximo timestamp y el valor anterior es mayor que 0 y el valor actual es 0
            if idx_row % n_timestamp == 0 \
                    and new_data.iloc[idx_row - 1, wifi] > 0 \
                    and new_data.iloc[idx_row, wifi] == 0:
                new_data.iloc[idx_row, wifi] = new_data.iloc[idx_row - 1, wifi]  # Rellenamos con el valor anterior

    return new_data


def interpolacion_pixel_proximo(data: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Aplica una interpolación a los datos de WiFi, de forma que si hay un hueco en el timestamp, se rellena con el valor anterior si y solo si el valor anterior y posterior son iguales

    Parameters:
    -----------
    data: pd.DataFrame
        pd.DataFrame de los datos correspondientes al WiFi

    threshold: int
        Número de segundos sin recogida de datos consecutivos para considerar si rechazar la interpolación

    Returns:
    --------
    interpolated_data: pd.DataFrame
        pd.DataFrame de los datos interpolados

    Example:
    --------

    Lectura del conjunto de datos

    >>> data = pd.read_csv(f"{constants.data.train.FINAL_PATH}/groundtruth.csv")

    Aplicación de la interpolación

    >>> interpolated_data = interpolacion_pixel_proximo(data, threshold=30)
    """
    n_timestamp = data["AppTimestamp(s)"].unique().max() # Maximo timestamp de cada label
    coords_unique = data.drop_duplicates(["Longitude", "Latitude"])[["Longitude", "Latitude"]].reset_index() # Coordenadas únicas
    interpolated_data = pd.DataFrame(columns=data.columns) # Dataframe vacío

    # Recorremos cada coordenada única
    for idx_coord, [x, y] in tqdm(enumerate(zip(coords_unique["Longitude"], coords_unique["Latitude"]))):
        query = data[(data["Longitude"] == x) & (data["Latitude"] == y)].reset_index() # Filtramos por coordenada

        # Recorremos cada AP
        for n_ap, ap in enumerate(constants.aps):

            # Recorremos cada segundo de cada AP
            for t in range(n_timestamp + 1):
                if query[ap].sum() == 0: # Si todos los valores son 0, salimos del bucle
                    break
                if query[ap].iloc[t] == 0: # Si el valor es 0
                    if t == 0: # Si es el primer segundo
                        right_values = query.loc[(t + 1):min(n_timestamp, t + threshold), ap].values # Valores a la derecha
                        right_values_distinct_zero = [idx for idx, x in enumerate(right_values) if x > 0] # Indices de los valores distintos de 0
                        idx_min_right = np.min(right_values_distinct_zero) if len(right_values_distinct_zero) > 0 else 0 # Índice del valor mínimo
                        query[ap].iloc[t] = right_values[idx_min_right] # Rellenamos con el primer valor derecho distinto de 0
                    elif t == n_timestamp: # Si es el último segundo
                        left_values = query.loc[max(0, t - threshold):(t - 1), ap].values # Valores a la izquierda
                        left_values_distinct_zero = [idx for idx, x in enumerate(left_values) if x > 0] # Indices de los valores distintos de 0
                        idx_max_left = np.max(left_values_distinct_zero) if len(left_values_distinct_zero) > 0 else 0 # Índice del valor máximo
                        query[ap].iloc[t] = left_values[idx_max_left] # Rellenamos con el último valor izquierdo distinto de 0
                    else:

                        left_values = query.loc[max(0, t - threshold):(t - 1), ap].values # Valores a la izquierda
                        right_values = query.loc[(t + 1):min(n_timestamp + 1, t + threshold + 1), ap].values # Valores a la derecha

                        left_values_distinct_zero = [idx for idx, x in enumerate(left_values) if x > 0] # Indices de los valores distintos de 0
                        right_values_distinct_zero = [idx for idx, x in enumerate(right_values) if x > 0] # Indices de los valores distintos de 0

                        idx_max_left = np.max(left_values_distinct_zero) if len(left_values_distinct_zero) > 0 else 0 # Máximo índice en la izquierda
                        idx_min_right = np.min(right_values_distinct_zero) if len(right_values_distinct_zero) > 0 else 0 # Mínimo índice en la derecha

                        if left_values[idx_max_left] == right_values[idx_min_right]: # Si los valores son iguales
                            query[ap].iloc[t] = left_values[idx_max_left] # Rellenamos con el valor izquierdo

        interpolated_data = interpolated_data.append(query) # Añadimos al dataframe

    return interpolated_data


def read_checkpoint(checkpoint_path: str) -> pd.DataFrame:
    """
    Lee los datos de un checkpoint y los concatena en un único dataframe

    Parameters:
    -----------
    checkpoint_path: str
        Path del checkpoint

    Returns:
    --------
    df: pd.DataFrame
        pd.DataFrame con los datos concatenados

    Example:
    --------
    Ruta al checkpoint del acelerómetro:

    >>> CHECKPOINT_ACCELEROMETER_PATH = "data/train/checkpoint_groundtruth/Accelerometer"

    Ruta al checkpoint del giroscopio:

    >>> CHECKPOINT_GYROSCOPE_PATH = "data/train/checkpoint_groundtruth/Gyroscope"

    Ruta al checkpoint del magnetómetro:

    >>> CHECKPOINT_MAGNETOMETER_PATH = "data/train/checkpoint_groundtruth/Magnetometer"

    Ruta al checkpoint del WiFi:

    >>> CHECKPOINT_WIFI_PATH = "data/train/checkpoint_groundtruth/Wifi"

    Lectura de los datos del acelerómetro:

    >>> accelerometer = read_checkpoint(CHECKPOINT_DATA_PATH)

    Lectura de los datos del giroscopio:

    >>> gyroscope = read_checkpoint(CHECKPOINT_GYROSCOPE_PATH)

    Lectura de los datos del magnetómetro:

    >>> magnetometer = read_checkpoint(CHECKPOINT_MAGNETOMETER_PATH)

    Lectura de los datos del WiFi:

    >>> wifi = read_checkpoint(CHECKPOINT_WIFI_PATH)
    """

    df = pd.DataFrame()
    for path in glob.glob(os.path.join(checkpoint_path, "*.csv")):
        df = pd.concat((df, pd.read_csv(path)))
    return df
