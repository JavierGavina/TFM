import os, tqdm

import pandas as pd
from src.constants import constants
from src.preprocess import correctMetrics, correctWifiFP, read_checkpoint

# Definición de las constantes de los directorios
CHECKPOINT_DATA_PATH = constants.data.train.CHECKPOINT_DATA_PATH
ACCELEROMETER_CHECKPOINT = f"{CHECKPOINT_DATA_PATH}/Accelerometer"
MAGNETOMETER_CHECKPOINT = f"{CHECKPOINT_DATA_PATH}/Magnetometer"
GYROSCOPE_CHECKPOINT = f"{CHECKPOINT_DATA_PATH}/Gyroscope"
WIFI_CHECKPOINT = f"{CHECKPOINT_DATA_PATH}/Wifi"

# Cogemos los ssids que han aparecido en más de una recogida de datos
lista_ssid_candidatos = constants.aps

# Tiempo maximo de muestreo por label
t_max_sampling = constants.T_MAX_SAMPLING

# Diccionario de labels a metros
labels_dictionary_meters = constants.labels_dictionary_meters


def main():
    os.makedirs(CHECKPOINT_DATA_PATH, exist_ok=True)  # Este código lo ejecutamos una vez y lo comentamos
    os.makedirs(ACCELEROMETER_CHECKPOINT, exist_ok=True)  # Este código lo ejecutamos una vez y lo comentamos
    os.makedirs(GYROSCOPE_CHECKPOINT, exist_ok=True)  # Este código lo ejecutamos una vez y lo comentamos
    os.makedirs(MAGNETOMETER_CHECKPOINT, exist_ok=True)  # Este código lo ejecutamos una vez y lo comentamos
    os.makedirs(WIFI_CHECKPOINT, exist_ok=True)  # Este código lo ejecutamos una vez y lo comentamos

    # Lectura del dataset de cada label
    for label, position in tqdm.tqdm(labels_dictionary_meters.items()):
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

            # Lectura de los datos de cada label
            with open(f'{constants.data.train.DATA_DIR}/label_{label}.txt', 'r') as fich:
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

    os.makedirs(constants.data.train.MID_PATH, exist_ok=True)  # Creamos el directorio si no existe

    accelerometer_data = correctMetrics(data=accelerometer_data,
                                        columns_to_correct=constants.accelerometer_cols,
                                        t_max_sampling=t_max_sampling,
                                        dict_labels_to_meters=labels_dictionary_meters)  # Corregimos el acelerómetro

    magnetometer_data = correctMetrics(data=magnetometer_data,
                                       columns_to_correct=constants.magnetometer_cols,
                                       t_max_sampling=t_max_sampling,
                                       dict_labels_to_meters=labels_dictionary_meters)  # Corregimos el magnetómetro

    gyroscope_data = correctMetrics(data=gyroscope_data,
                                    columns_to_correct=constants.gyroscope_cols,
                                    t_max_sampling=t_max_sampling,
                                    dict_labels_to_meters=labels_dictionary_meters)  # Corregimos el giroscopio

    wifi_data = correctWifiFP(wifi_data=wifi_data,
                              t_max_sampling=t_max_sampling,
                              dict_labels_to_meters=labels_dictionary_meters)  # Corregimos el WiFi

    # Guardamos los datos corregidos
    accelerometer_data.to_csv(f"{constants.data.train.MID_PATH}/accelerometer.csv", index=False)
    magnetometer_data.to_csv(f"{constants.data.train.MID_PATH}/magnetometer.csv", index=False)
    gyroscope_data.to_csv(f"{constants.data.train.MID_PATH}/gyroscope.csv", index=False)
    wifi_data.to_csv(f"{constants.data.train.MID_PATH}/wifi.csv", index=False)

    '''
    Juntamos todos los datos y los guardamos en el path del acelerómetro
    '''
    os.makedirs(constants.data.train.FINAL_PATH, exist_ok=True)  # Creamos el directorio si no existe
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
    joined_data.to_csv(f"{constants.data.train.FINAL_PATH}/groundtruth.csv", index=False)


if __name__ == "__main__":
    main()
