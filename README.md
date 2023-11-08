# cGAN data augmentation for indoor positioning

En la era actual de la conectividad inalámbrica, el uso de tecnologías basadas en WiFi se ha convertido en un componente esencial de nuestras vidas cotidianas. La ubicación y la geolocalización son factores críticos para una amplia gama de aplicaciones, desde la navegación en interiores hasta la optimización de la logística en almacenes inteligentes. Una de las técnicas más comunes para estimar la posición en entornos interiores se basa en el uso del "Fingerprinting" del WiFi, que implica la creación de una base de datos de señales RSSI (Received Signal Strength Indicator) en puntos de referencia conocidos. Sin embargo, la creación y el mantenimiento de estas bases de datos puede ser un proceso costoso y laborioso.

El presente Trabajo de Fin de Máster (TFM) se enfoca en abordar este desafío al aplicar Generative Adversarial Networks condicionales (cGANs) para aumentar una base de datos de señales RSSI obtenidas a través de la aplicación "get_sensordata". Esta aplicación recopila información de señales WiFi en entornos específicos y constituye una herramienta valiosa para la recopilación de datos de entrenamiento para sistemas de posicionamiento basados en huellas digitales de WiFi.

El objetivo fundamental de este TFM es mejorar las técnicas de estimación de posición en entornos interiores mediante la expansión de la base de datos de señales RSSI a través de la generación sintética de datos utilizando cGANs. La aplicación de GANs condicionales permitirá generar datos RSSI adicionales que se asemejen a los recopilados en el mundo real, lo que a su vez mejorará la precisión y la robustez de los sistemas de posicionamiento basados en huellas digitales de WiFi.

Este trabajo se estructura en torno a la investigación, el diseño, la implementación y la evaluación de un sistema que integra cGANs para aumentar la base de datos de señales RSSI y, finalmente, mejorar las técnicas de estimación de posición en interiores. Se llevará a cabo una revisión exhaustiva de la literatura relacionada, se presentará una metodología de trabajo detallada y se realizarán experimentos para evaluar la eficacia de la técnica propuesta.

Con el crecimiento continuo de la Internet de las cosas (IoT) y la necesidad de sistemas de posicionamiento precisos en entornos interiores, este TFM se presenta como una contribución significativa al campo de la geolocalización basada en WiFi, al abordar la problemática de la expansión de las bases de datos RSSI de manera innovadora y efectiva.

A lo largo de las próximas secciones, se detallarán los aspectos metodológicos, los resultados obtenidos y las conclusiones derivadas de esta investigación.

## Preprocesamiento de los datos

Los datos obtenidos con la aplicación de **get_sensordata** se encuentran en el directorio *data/raw_groundtruth*.

El script **process_groundTruth.py** se encarga de procesar los datos en bruto, limpiarlos y exportalos en la ruta *data/final_groundtruth*

## Carga de los datos

```python
from src import dataloader
from src import constants
X, y = dataloader.DataLoader(data_dir=f"../{constants.directories.FINAL_PATH}/groundtruth.csv",
                             aps_list=constants.aps, batch_size=30, step_size=5,
                             size_reference_point_map=300,
                             return_axis_coords=False)()
```

## Entrenamiento del modelo

Para entrenar el modelo, ya se encuentra definido el tipo de modelo cGAN a utilizar dentro de **models**. Además, se encuentran definidos una serie de **callbacks**
 customizados para utilizar en el proceso de entrenamiento de la cGAN dentro del mismo directorio. Entre ellos tenemos definido:

<ul>
    <li><b>SaveImageTraining:</b> Guarda la imagen real, la imagen generada y el histograma de la imagen generada tras cada época </li>
    <li><b>LoggingCheckpointTraining:</b> Guarda el modelo en formato .h5 cada 10 épocas</li>
</ul>

```python
import tf

# Importación del modelo cGAN
from models.cGAN_300_300 import cGAN
# uso de callbacks definidas para la cGAN
from models.callbacks import SaveImageTraining, LoggingCheckpointTraining

# Definición del discriminador y el generador
def define_discriminator():
    ....

def define_generator(latent_dim):
    ....

discriminator = define_discriminator()
generator = define_generator(latent_dim=100)

# definición del dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(1000).batch(8)

# definición de callbacks
path_images_training = f"../{constants.models.cgan_300}/training_images"
path_checkpoints = f"../{constants.models.cgan_300}/checkpoints"
save_image = SaveImageTraining(X, y, save_dir=path_images_training)
save_model = LoggingCheckpointTraining(save_dir=path_checkpoints)

# Entrenamiento del modelo
cgan = cGAN(generator=generator, discriminator=discriminator)
cgan.fit(dataset, epochs=50, callbacks = [save_image, save_model])
```