class Directories:
    """
    Clase que contiene las constantes de los directorios del proyecto

    Attributes:
    ___________
        DATA_DIR: str
            La dirección de los datos en bruto (formato .txt)
        CHECKPOINT_DATA_PATH: str
            La dirección de los datos del checkpoint de cada métrica y wifi
        MID_PATH: str
            La dirección de salida de cada dataset (metricas y wifi)
        FINAL_PATH: str
            Datos unidos
    """
    # Definición de las constantes
    DATA_DIR = "data/raw_groundtruth"  # La dirección de los datos extendida
    CHECKPOINT_DATA_PATH = "data/checkpoint_groundtruth"
    MID_PATH = "data/mid_groundtruth"  # La dirección de salida de cada dataset (metricas y wifi)
    FINAL_PATH = "data/final_groundtruth"  # Datos unidos


class Architectures:
    """
    Clase que contiene las constantes de los directorios de las arquitecturas de los modelos
    """
    arquitectures = "outputs/model_architecture"
    cgan_300 = f"{arquitectures}/cGAN_300_300"
    cgan_28 = f"{arquitectures}/cGAN_28_28"


class Models:
    """
    Clase que contiene las constantes de los directorios de los modelos del proyecto
    """
    training = "outputs/process_training"
    cgan_300 = "outputs/process_training/cGAN_300_300"
    cgan_28 = "outputs/process_training/cGAN_28_28"


class RPMAP:
    PATH_RPMAP = "outputs/RPMap"
    rpmap_300_overlapping = f"{PATH_RPMAP}/rpmap_300_overlapping"
    rpmap_300_sinOverlapping = f"{PATH_RPMAP}/rpmap_300_sinOverlapping"
    rpmap_28_overlapping = f"{PATH_RPMAP}/rpmap_28_overlapping"


class Outputs:
    """
    Clase que contiene las constantes de los directorios de los outputs del proyecto

    Attributes:
    ___________
        PATH_OUTPUTS: str
            La dirección raíz de los outputs
        PATH_RPMAP: str
            La dirección de los mapas de referencia continua
    """
    PATH_OUTPUTS = "outputs"

    models = Models()
    rpmap = RPMAP()
    architectures = Architectures()


class constants:
    """
    Clase que contiene las constantes del proyecto

    Attributes:
    ___________
        dictionary_decoding: dict
            Diccionario que transforma el label a (longitud, latitud) en metros
        data: Directories
            Clase que contiene las constantes de los directorios del proyecto
        aps: list
            Lista de APs (wifi) a considerar para la generación del mapa de referencia continua
        magnetometer_cols: list
            Lista de columnas del magnetómetro
        accelerometer_cols: list
            Lista de columnas del acelerómetro
        gyroscope_cols: list
            Lista de columnas del giroscopio
        labels_dictionary_meters: dict
            Diccionario que transforma el label a (longitud, latitud) en metros
        outputs: Outputs
            Clase que contiene las constantes de los directorios de todas las salidas del proyecto
    """

    T_MAX_SAMPLING = 1140  # Número de segundos máximo de recogida de muestras por cada Reference Point

    dictionary_decoding = {
        0: "GEOTECWIFI03", 1: "480Invitados",
        2: "eduroam", 3: "wpen-uji",
        4: "lt1iot", 5: "cuatroochenta",
        6: "UJI"
    }

    data = Directories()

    aps = ['GEOTECWIFI03', '480Invitados', 'eduroam', 'wpen-uji', 'lt1iot', 'cuatroochenta', 'UJI']
    magnetometer_cols = ["Mag_X", "Mag_Y", "Mag_Z"]
    accelerometer_cols = ["Acc_X", "Acc_Y", "Acc_Z"]
    gyroscope_cols = ["Gyr_X", "Gyr_Y", "Gyr_Z"]

    # Diccionario que transforma el label a (longitud, latitud) en metros
    labels_dictionary_meters = {
        0: (0.6, 0), 1: (5.4, 0), 2: (9, 0),
        3: (9, 3), 4: (6, 3), 5: (3, 3),
        6: (0.6, 3), 7: (0.6, 4.8), 8: (3.6, 4.8),
        9: (6, 4.8), 10: (9, 4.8), 11: (9, 7.8),
        12: (6.6, 7.8), 13: (3, 7.8), 14: (0.6, 7.8),
        15: (0.6, 9.6), 16: (3, 9.6), 17: (4.8, 9.6),
        18: (8.4, 9.6), 19: (8.4, 12), 20: (8.4, 14.4),
        21: (3, 14.4), 22: (0, 14.4)
    }

    outputs = Outputs()
