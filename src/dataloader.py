import numpy as np
import pandas as pd
import warnings
from scipy.interpolate import griddata
import sys

sys.path.append("../")

from src.constants import constants
from process_train import interpolacionConcatenada


warnings.filterwarnings('ignore')


def preprocessGrid(grid_z0: np.ndarray) -> np.ndarray:
    """
    Esta función devuelve una matriz con los valores interpolados del RSS para toda latitud y longitud.
    Los valores interpolados se realizan dentro del espacio de muestreo comprendido entre (minLongitud, minLatitud) y (maxLongitud, maxLatitud). Por lo que todo punto de muestreo fuera de este espacio es fijado a np.nan por defecto.


    Esta función se encarga de rellenar los valores NaN con una Interpolación lineal unidimensional para puntos de muestra monotónicamente crecientes

    Parameters:
    -----------
    grid_z0: matriz con los valores interpolados del RSS para toda latitud y longitud

    Returns:
    --------
    grid_z0: matriz con los valores interpolados del RSS para toda latitud y longitud

    Examples:
    ---------

    >>> grid_z0 = np.array([[1, 2, 3], [np.nan, 5, 6], [7, 8, np.nan]])

    >>> preprocessGrid(grid_z0)

    >>> array([[1., 2., 3.],
               [4., 5., 6.],
               [7., 8., 6.]])
    """

    grid_z0 = grid_z0.ravel()
    nx = ny = np.int(np.sqrt(grid_z0.shape[0]))

    # Los valores por encima de 1 y por debajo de 0 carecen de sentido, se fijan a 1 y 0 respectivamente
    grid_z0[grid_z0 < 0] = 0
    grid_z0[grid_z0 > 1] = 1

    # Encontrar los índices de los valores conocidos
    known_indexes = np.where(~np.isnan(grid_z0))[0]

    # Encontrar los índices de los valores que desea interpolar
    interp_indexes = np.where(np.isnan(grid_z0))[0]

    # Interpolar los valores NaN con otros valores
    interp_values = np.interp(interp_indexes, known_indexes, grid_z0[known_indexes])

    # Reemplazar los valores NaN por los valores interpolados
    grid_z0[np.isnan(grid_z0)] = interp_values

    return grid_z0.reshape((nx, ny))


def parse_windows(n_max: int, window_size: int, step: int):
    """
    Esta función devuelve una lista de ventanas de tamaño window_size y step step.

    Parameters:
    -----------
    n_max: int
        número máximo a generar en la lista de ventanas

    window_size: int
        tamaño de la ventana

    step: int
        tamaño del paso

    Returns:
    --------
    list
        Lista de ventanas (tuplas) con índices (start, end) de cada ventana generada en base a la longitud de la serie temporal, el tamaño de la ventana y el tamaño del paso

    Examples:
    ---------
    >>> parse_windows(n_max=10, window_size=5, step=2)
    >>> [(0, 5), (2, 7), (4, 9), (6, 10)]

    >>> parse_windows(n_max=30, window_size=10, step=5)
    >>> [(0, 10), (5, 15), (10, 20), (15, 25), (20, 30)]

    """
    return [(i, min(i + window_size, n_max)) for i in range(0, n_max, step) if i + window_size <= n_max]


def referencePointMap(dataframe: pd.DataFrame, aps_list: list, batch_size: int = 30, step_size: int = 5,
                      size_reference_point_map: int = 300,
                      return_axis_coords: bool = False):
    """
    Esta función devuelve una matriz con los valores interpolados del RSS de cada AP (wifi) para toda latitud y longitud en el espacio de muestreo.

    Parameters:
    -----------
    dataframe: pd.DataFrame
        DataFrame con los valores de RSS de cada AP (wifi) para cada latitud y longitud anotada en un Reference Point (RP)

    aps_list: list
        Lista de APs (wifi) a considerar para la generación del mapa de referencia continua

    batch_size: int = 30
        Tamaño de la ventana de tiempo para la generación de cada mapa de referencia continua (número de segundos a considerar para calcular la media de RSS en cada RP)

    step_size: int = 5
        Número de segundos que se desplaza la ventana de tiempo para la generación de cada mapa de referencia continua. Si no queremos overlapping (entrecruzamiento de ventanas), tenemos que asignar el mismo valor que batch_size

    size_reference_point_map: int
        Tamaño del mapa de referencia continua (número de RPs muestreadas en cada dimensión)

    return_axis_coords: bool
        Si es True, devuelve las coordenadas x e y del mapa de referencia continua. Si es False, devuelve únicamente el mapa de referencia continua y las etiquetas de los APs (wifi)

    Returns:
    --------
    reference_point_map: np.ndarray
        Matriz con los valores interpolados del RSS de cada AP (wifi) para toda latitud y longitud en el espacio de muestreo

    APLabel: np.ndarray
        Etiquetas de los APs (wifi) para cada mapa de referencia continua

    -----------------------------------------
    En caso en que return_axis_coords sea True:
    -----------------------------------------
    x_g: np.ndarray
        Coordenadas longitud continuas del mapa de referencia continua

    y_g: np.ndarray
        Coordenadas latitud continuas del mapa de referencia continua

    Examples:
    ---------

    Lectura de datos

    >>> dataframe = pd.DataFrame({
            "AppTimestamp(s)": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "Longitude": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "Latitude": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "GEOTECWIFI03": [0, 0, 0.2, 0, 0.2, 0.8, 1, 0.8, 1, 1],
            "eduroam": [1, 0.9, 1, 0.9, 0.8, 0.2, 0.1, 0.2, 0, 0],
        })

    Lista de aps

    >>> aps_list = ["GEOTECWIFI03", "eduroam"]

    Generación de mapa de referencia continua

    >>> reference_point_map, APLabel = referencePointMap(dataframe, aps_list=aps_list, batch_size=5, step_size=2, size_reference_point_map=2)
    """

    nx = ny = size_reference_point_map
    t_max = dataframe["AppTimestamp(s)"].max()
    samples_per_RP = int((t_max - batch_size) / step_size) + 1
    RPMap = np.zeros((samples_per_RP * len(aps_list), nx, ny))
    APLabel = []
    combinaciones = parse_windows(t_max, batch_size, step_size)

    for n_ap in range(0, len(aps_list) * samples_per_RP, samples_per_RP):
        ap = aps_list[n_ap // samples_per_RP]
        for batch, (start, end) in enumerate(combinaciones):
            aux = dataframe[(dataframe["AppTimestamp(s)"] >= start) & (dataframe["AppTimestamp(s)"] < end)]
            aux = aux.groupby(["Longitude", "Latitude"]).mean()[ap].reset_index()
            miny, maxy = min(aux['Latitude']), max(aux['Latitude'])
            minx, maxx = min(aux['Longitude']), max(aux['Longitude'])
            grdi_x = np.linspace(minx, maxx, num=nx, endpoint=False)  # Coordenadas x discretas
            grdi_y = np.linspace(miny, maxy, num=ny, endpoint=False)  # Coordenadas y discretas
            yg, xg = np.meshgrid(grdi_y, grdi_x, indexing='ij')  # Coordenadas x e y continuas
            x_g = xg.ravel()  # Coordenadas x continuas
            y_g = yg.ravel()  # Coordenadas y continuas
            aux2 = aux.drop([ap], 1)
            points = np.array(aux2)
            values = np.array(aux[ap])
            grid_z0 = griddata(points, values, (x_g, y_g), method='cubic')

            RPMap[batch + n_ap, :, :] = preprocessGrid(grid_z0)
            APLabel.append(ap)

    APLabel = np.array(APLabel)
    if return_axis_coords:
        return RPMap, APLabel, x_g, y_g

    return RPMap, APLabel


def labelEncoding(labels: np.ndarray) -> np.ndarray:
    """
    Esta función devuelve un array con los valores de las etiquetas codificadas en números enteros.

    Parameters:
    -----------
    labels: np.ndarray
        Array con los valores de las etiquetas a codificar

    Returns:
    --------
        numericLabels: np.ndarray
            Array con los valores de las etiquetas codificadas en números enteros

    Examples:
    ---------

    >>> labels = np.array(["GEOTECWIFI03", "480Invitados", "eduroam", "wpen-uji", "lt1iot", "cuatroochenta", "UJI"])

    >>> labelEncoding(labels)

    >>> array([0, 1, 2, 3, 4, 5, 6])
    """
    numericLabels = labels.copy()
    numericLabels[numericLabels == constants.aps[0]] = 0
    numericLabels[numericLabels == constants.aps[1]] = 1
    numericLabels[numericLabels == constants.aps[2]] = 2
    numericLabels[numericLabels == constants.aps[3]] = 3
    numericLabels[numericLabels == constants.aps[4]] = 4
    numericLabels[numericLabels == constants.aps[5]] = 5
    numericLabels[numericLabels == constants.aps[6]] = 6
    return numericLabels.astype(int)


def labelDecoding(labels: np.ndarray) -> np.ndarray:
    """
    Esta función devuelve un array con los valores de las etiquetas decodificadas en strings.

    Parameters:
    -----------
    labels: np.ndarray
        Array con los valores de las etiquetas a decodificar

    Returns:
    --------
    categoricLabels: np.ndarray
        Array con los valores de las etiquetas decodificadas en strings

    Examples:
    ---------

    >>> labels = np.array([0, 1, 2, 3, 4, 5, 6])

    >>> labelDecoding(labels)

    >>> array(["GEOTECWIFI03", "480Invitados", "eduroam", "wpen-uji", "lt1iot", "cuatroochenta", "UJI"])
    """

    categoricLabels = labels.copy()
    if len(categoricLabels.shape) == 2:
        categoricLabels = categoricLabels.reshape(categoricLabels.shape[0], )

    categoricLabels = pd.Series(categoricLabels).map(lambda x: constants.dictionary_decoding[x]).to_numpy()
    return categoricLabels


class DataLoader:
    """
    Esta clase se encarga de cargar los datos de los mapas de referencia continua para cada AP (wifi) y sus etiquetas.

    Attributes:
    -----------
    data_dir: str
        Ruta del archivo csv con los datos de los mapas de referencia continua para cada AP (wifi) y sus etiquetas
    aps_list: list
        Lista de APs (wifi) a considerar para la generación del mapa de referencia continua
    batch_size: int = 30
        Tamaño de la ventana de tiempo para la generación de cada mapa de referencia continua (número de segundos a considerar para calcular la media de RSS en cada RP)
    step_size: int = 5
        Número de segundos que se desplaza la ventana de tiempo para la generación de cada mapa de referencia continua. Si no queremos overlapping (entrecruzamiento de ventanas), tenemos que asignar el mismo valor que batch_size
    size_reference_point_map: int = 300
        Tamaño del mapa de referencia continua (número de RPs muestreadas en cada dimensión)
    return_axis_coords: bool = False
        Si es True, devuelve las coordenadas x e y del mapa de referencia continua. Si es False, devuelve únicamente el mapa de referencia continua y las etiquetas de los APs (wifi)

    Methods:
    --------
    __getData(data_dir: str) -> pd.DataFrame:
        Esta función devuelve un DataFrame con los datos de los mapas de referencia continua para cada AP (wifi) y sus etiquetas

    __call__(*args, **kwargs) -> np.ndarray:
        Esta función devuelve los mapas de referencia continua para cada AP (wifi) y sus etiquetas

    Examples:
    ---------

    Con overlapping

    >>> dataloader = DataLoader(data_dir=f"{constants.data.FINAL_PATH}/groundtruth.csv",
                                aps_list=constants.aps, batch_size=30, step_size=5,
                                size_reference_point_map=300,
                                return_axis_coords=False)

    >>> X, y, _ = dataloader()

    Sin overlapping

    >>> dataloader = DataLoader(data_dir=f"{constants.data.FINAL_PATH}/groundtruth.csv",
                                aps_list=constants.aps, batch_size=30, step_size=30,
                                size_reference_point_map=300,
                                return_axis_coords=False)

    >>> X, y, _ = dataloader()

    Devolver coordenadas x e y

    >>> dataloader = DataLoader(data_dir=f"{constants.data.FINAL_PATH}/groundtruth.csv",
                                aps_list=constants.aps, batch_size=30, step_size=5,
                                size_reference_point_map=300,
                                return_axis_coords=True)

    >>> X, y, [x_coords, y_coords] = dataloader()

    """
    def __init__(self, data_dir: str, aps_list: list, batch_size: int = 30, step_size: int = 5,
                 size_reference_point_map: int = 300,
                 return_axis_coords: bool = False):
        self.groundtruth = self.__getData(data_dir)
        self.aps_list = aps_list
        self.batch_size = batch_size
        self.step_size = step_size
        self.size_reference_point_map = size_reference_point_map
        self.return_axis_coords = return_axis_coords

    @staticmethod
    def __getData(data_dir):
        groundtruth = pd.read_csv(data_dir)
        new_columns = ["AppTimestamp(s)"] + constants.aps + constants.accelerometer_cols + \
                      constants.gyroscope_cols + constants.magnetometer_cols + ["Latitude", "Longitude", "Label"]

        groundtruth = groundtruth[new_columns]
        groundtruth_interpolated = interpolacionConcatenada(groundtruth)
        return groundtruth_interpolated

    def __call__(self, *args, **kwargs):
        [X, y, *coords] = referencePointMap(self.groundtruth, aps_list=self.aps_list,
                                            batch_size=self.batch_size,
                                            step_size=self.step_size,
                                            size_reference_point_map=self.size_reference_point_map,
                                            return_axis_coords=self.return_axis_coords)
        X, y = np.expand_dims(X, axis=-1), np.expand_dims(y, axis=-1)
        return X, y, coords
