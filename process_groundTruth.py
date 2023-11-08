import os, tqdm

import numpy as np
import pandas as pd
import glob
from src.constants import constants

ACCELEROMETER_CHECKPOINT = f"{constants.directories.CHECKPOINT_DATA_PATH}/Accelerometer"
MAGNETOMETER_CHECKPOINT = f"{constants.directories.CHECKPOINT_DATA_PATH}/Magnetomerer"
GYROSCOPE_CHECKPOINT = f"{constants.directories.CHECKPOINT_DATA_PATH}/Gyroscope"
WIFI_CHECKPOINT = f"{constants.directories.CHECKPOINT_DATA_PATH}/Wifi"

T_MAX_SAMPLING = 1140  # Número de segundos máximo de recogida de muestras por cada Reference Point

# Cogemos los ssids que han aparecido en más de una recogida de datos
lista_ssid_candidatos = constants.aps


def correctWifiFP(wifi_data: pd.DataFrame) -> pd.DataFrame:
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

    Returns:
    --------
    wifi_corrected: pd.DataFrame
        pd.DataFrame de los datos corregidos

    Example:
    --------
    >>> wifi_corrected = correctWifiFP(wifi_data)
    """

    # Cogemos los datos de las balizas que han aparecido en más de una recogida de datos
    aux = wifi_data.query(f'Name_SSID in {lista_ssid_candidatos}')

    # Creamos un dataframe con todos los intervalos de tiempo y todas las balizas
    labels, intervalos_tiempo, ssids = [], [], []
    for lab in range(23):
        for ts in range(0, T_MAX_SAMPLING + 1, 1):
            for ssid in lista_ssid_candidatos:
                labels.append(lab)
                intervalos_tiempo.append(ts)
                ssids.append(ssid)
    df_intervalos = pd.DataFrame(
        {'Label': labels, 'AppTimestamp(s)': intervalos_tiempo, 'Name_SSID': ssids})  # Dataframe de intervalos

    # Ajuste de frecuencia muestral a 1 muestra / segundo
    min_global = aux.RSS.min() - 1  # Valor mínimo de RSS
    max_global = np.abs(np.max(aux.RSS.max()))  # Valor máximo de RSS
    aux["AppTimestamp(s)"] = aux["AppTimestamp(s)"].round()  # Redondeamos el timestamp a 0 decimales
    aux = aux.groupby(["Label", "AppTimestamp(s)", "Name_SSID"]).mean()[
        "RSS"].reset_index()  # Agrupamos por label, timestamp y ssid
    aux_corrected = pd.merge(df_intervalos, aux, on=["Label", "AppTimestamp(s)", "Name_SSID"],
                             how="left")  # Unimos con el dataframe de intervalos
    aux_corrected["RSS"] = aux_corrected.RSS.replace(np.nan,
                                                     min_global)  # Reemplazamos los valores ausentes por el mínimo global
    aux_corrected = aux_corrected.pivot(index=['Label', 'AppTimestamp(s)'], columns='Name_SSID')[
        "RSS"].reset_index()  # Pivotamos el dataframe
    aux_corrected.iloc[:, 2:] = (aux_corrected.iloc[:,
                                 2:] - min_global) / max_global  # Escalamos los valores entre 0 y 1
    aux_corrected[["Longitude", "Latitude"]] = [constants.labels_dictionary_meters[x] for x in
                                                aux_corrected.Label]  # Añadimos la longitud y latitud
    orden_wifi_columnas = ["AppTimestamp(s)"] + \
                          [x for x in aux_corrected.columns if
                           x not in ["AppTimestamp(s)", "Latitude", "Longitude", "Label"]] + \
                          ["Latitude", "Longitude", "Label"]
    wifi_corrected = aux_corrected[orden_wifi_columnas]  # Ordenamos las columnas
    return wifi_corrected


def correctMetrics(data: pd.DataFrame, columns_to_correct: list) -> pd.DataFrame:
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

    intervalos_tiempo = np.arange(0, T_MAX_SAMPLING + 1, 1)
    df_intervalos = pd.DataFrame({'AppTimestamp(s)': intervalos_tiempo})
    data_corrected = pd.DataFrame(
        columns=["AppTimestamp(s)"] + columns_to_correct +
                ["Latitude", "Longitude", "Label"])
    for label, position in constants.labels_dictionary_meters.items():
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

            # Si es multiplo del maximo timestamp y el valor anterior es mayor que 0 y el valor actual es 0
            if idx_row % n_timestamp == 0 \
                    and new_data.iloc[idx_row - 1, wifi] > 0 \
                    and new_data.iloc[idx_row, wifi] == 0:
                new_data.iloc[idx_row, wifi] = new_data.iloc[idx_row - 1, wifi]  # Rellenamos con el valor anterior

    return new_data


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

    >>> CHECKPOINT_ACCELEROMETER_PATH = "data/checkpoint_groundtruth/Accelerometer"

    Ruta al checkpoint del giroscopio:

    >>> CHECKPOINT_GYROSCOPE_PATH = "data/checkpoint_groundtruth/Gyroscope"

    Ruta al checkpoint del magnetómetro:

    >>> CHECKPOINT_MAGNETOMETER_PATH = "data/checkpoint_groundtruth/Magnetometer"

    Ruta al checkpoint del WiFi:

    >>> CHECKPOINT_WIFI_PATH = "data/checkpoint_groundtruth/Wifi"

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
    for dir in glob.glob(os.path.join(checkpoint_path, "*.csv")):
        df = pd.concat((df, pd.read_csv(dir)))
    return df


def main():
    # os.makedirs(CHECKPOINT_DATA_PATH, exist_ok=True) # Este código lo ejecutamos una vez y lo comentamos
    # os.makedirs(ACCELEROMETER_CHECKPOINT, exist_ok=True) # Este código lo ejecutamos una vez y lo comentamos
    # os.makedirs(GYROSCOPE_CHECKPOINT, exist_ok=True) # Este código lo ejecutamos una vez y lo comentamos
    # os.makedirs(MAGNETOMETER_CHECKPOINT, exist_ok=True) # Este código lo ejecutamos una vez y lo comentamos
    # os.makedirs(WIFI_CHECKPOINT, exist_ok=True) # Este código lo ejecutamos una vez y lo comentamos

    # Lectura del dataset de cada label
    for label, position in tqdm.tqdm(constants.labels_dictionary_meters.items()):
        # Si no existe el checkpoint con el label, lo creamos
        if str(label) not in ", ".join(os.listdir(ACCELEROMETER_CHECKPOINT)):
            # Inicialización datasets métricas y WiFi
            accelerometer_data = pd.DataFrame(
                columns=["AppTimestamp(s)", "SensorTimestamp(s)", "Acc_X", "Acc_Y", "Acc_Z", "Label", "Latitude",
                         "Longitude"])
            gyroscope_data = pd.DataFrame(
                columns=["AppTimestamp(s)", "SensorTimestamp(s)", "Gyr_X", "Gyr_Y", "Gyr_Z", "Label", "Latitude",
                         "Longitude"])
            magnetometer_data = pd.DataFrame(
                columns=["AppTimestamp(s)", "SensorTimestamp(s)", "Mag_X", "Mag_Y", "Mag_Z", "Label", "Latitude",
                         "Longitude"])
            wifi_data = pd.DataFrame(
                columns=['AppTimestamp(s)', 'SensorTimeStamp(s)', 'Name_SSID', 'MAC_BSSID', 'RSS'])

            with open(f'{constants.Directories.DATA_DIR}/label_{label}.txt', 'r') as fich:
                for linea in fich.readlines():  # Leemos línea
                    texto = linea.rstrip("\n").split(";")  # Quitamos saltos de línea y separamos por ";"

                    if texto[0] == "ACCE":  # Si el primer elemento es ACCE, añadimos datos al acelerómetro
                        accelerometer_data = pd.concat([accelerometer_data,
                                                        pd.DataFrame({
                                                            "AppTimestamp(s)": [float(texto[1])],
                                                            "SensorTimestamp(s)": [float(texto[2])],
                                                            "Acc_X": [float(texto[3])],
                                                            "Acc_Y": [float(texto[4])],
                                                            "Acc_Z": [float(texto[5])],
                                                            "Label": [label],
                                                            "Latitude": [position[1]],
                                                            "Longitude": [position[0]]
                                                        })], ignore_index=True)
                    if texto[0] == "GYRO":  # Si el primer elemento es GYRO, añadimos datos al giroscopio
                        gyroscope_data = pd.concat([gyroscope_data,
                                                    pd.DataFrame({
                                                        "AppTimestamp(s)": [float(texto[1])],
                                                        "SensorTimestamp(s)": [float(texto[2])],
                                                        "Gyr_X": [float(texto[3])],
                                                        "Gyr_Y": [float(texto[4])],
                                                        "Gyr_Z": [float(texto[5])],
                                                        "Label": [label],
                                                        "Latitude": [position[1]],
                                                        "Longitude": [position[0]]
                                                    })], ignore_index=True)

                    if texto[0] == "MAGN":  # Si el primer elemento es ACCE, añadimos datos al acelerómetro
                        magnetometer_data = pd.concat([magnetometer_data,
                                                       pd.DataFrame({
                                                           "AppTimestamp(s)": [float(texto[1])],
                                                           "SensorTimestamp(s)": [float(texto[2])],
                                                           "Mag_X": [float(texto[3])],
                                                           "Mag_Y": [float(texto[4])],
                                                           "Mag_Z": [float(texto[5])],
                                                           "Label": [label],
                                                           "Latitude": [position[1]],
                                                           "Longitude": [position[0]]
                                                       })], ignore_index=True)
                    if texto[0] == "WIFI":  # Si el primer elemento es ACCE, añadimos datos al acelerómetro
                        wifi_data = pd.concat([wifi_data,
                                               pd.DataFrame({
                                                   "AppTimestamp(s)": [float(texto[1])],
                                                   "SensorTimeStamp(s)": [float(texto[2])],
                                                   "Name_SSID": [texto[3]],
                                                   "MAC_BSSID": [texto[4]],
                                                   "RSS": [float(texto[5])],
                                                   "Label": [label],
                                                   "Latitude": [position[1]],
                                                   "Longitude": [position[0]]
                                               })], ignore_index=True)

            # Guardamos los datos en el checkpoint
            accelerometer_data.to_csv(f"{ACCELEROMETER_CHECKPOINT}/Accelerometer_label_{label}.csv", index=False)
            gyroscope_data.to_csv(f"{GYROSCOPE_CHECKPOINT}/Gyroscope_label_{label}.csv", index=False)
            magnetometer_data.to_csv(f"{MAGNETOMETER_CHECKPOINT}/Magnetometer_label_{label}.csv", index=False)
            wifi_data.to_csv(f"{WIFI_CHECKPOINT}/Wifi_label_{label}.csv", index=False)

    # Lectura de los datos de cada checkpoint
    accelerometer_data = read_checkpoint(ACCELEROMETER_CHECKPOINT)
    magnetometer_data = read_checkpoint(MAGNETOMETER_CHECKPOINT)
    gyroscope_data = read_checkpoint(GYROSCOPE_CHECKPOINT)
    wifi_data = read_checkpoint(WIFI_CHECKPOINT)

    os.makedirs(constants.directories.MID_PATH, exist_ok=True)  # Creamos el directorio si no existe
    accelerometer_data = correctMetrics(data=accelerometer_data,
                                        columns_to_correct=constants.accelerometer_cols)  # Corregimos el acelerómetro
    magnetometer_data = correctMetrics(data=magnetometer_data,
                                       columns_to_correct=constants.magnetometer_cols)  # Corregimos el magnetómetro
    gyroscope_data = correctMetrics(data=gyroscope_data,
                                    columns_to_correct=constants.gyroscope_cols)  # Corregimos el giroscopio
    wifi_data = correctWifiFP(wifi_data=wifi_data)  # Corregimos el WiFi

    # Guardamos los datos corregidos
    accelerometer_data.to_csv(f"{constants.directories.MID_PATH}/accelerometer.csv", index=False)
    magnetometer_data.to_csv(f"{constants.directories.MID_PATH}/magnetometer.csv", index=False)
    gyroscope_data.to_csv(f"{constants.directories.MID_PATH}/gyroscope.csv", index=False)
    wifi_data.to_csv(f"{constants.directories.MID_PATH}/wifi.csv", index=False)

    '''
    Juntamos todos los datos y los guardamos en el path del acelerómetro
    '''
    os.makedirs(constants.directories.FINAL_PATH, exist_ok=True)  # Creamos el directorio si no existe
    cols_to_join = ["AppTimestamp(s)", "Latitude", "Longitude", "Label"]  # Columnas en común cada dataset
    order_of_columns = ["AppTimestamp(s)"] + \
                       constants.aps + constants.accelerometer_cols + \
                       constants.magnetometer_cols + constants.gyroscope_cols + \
                       ["Latitude", "Longitude", "Label"]  # Orden de las columnas

    # Juntamos los datos
    joined_data = pd.merge(wifi_data,
                           pd.merge(accelerometer_data,
                                    pd.merge(gyroscope_data, magnetometer_data, on=cols_to_join, how="left"),
                                    on=cols_to_join, how="left"),
                           on=cols_to_join, how="left")[order_of_columns]

    # guardamos el dataset final
    joined_data.to_csv(f"{constants.directories.FINAL_PATH}/groundtruth.csv", index=False)


if __name__ == "__main__":
    main()
