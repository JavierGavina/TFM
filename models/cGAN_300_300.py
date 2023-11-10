import os
import sys
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Embedding, Conv2D, \
    Conv2DTranspose, LeakyReLU, BatchNormalization


sys.path.append("../")
from src import dataloader
from src.constants import constants
from models.callbacks import SaveImageTraining, LoggingCheckpointTraining
import numpy as np


def define_discriminator(input_shape=(300, 300, 1), n_classes=7):
    # label input
    in_labels = Input(shape=(1,), name="label_discriminador")
    # Embedding for categorical input
    em = Embedding(n_classes, 50)(in_labels)
    # scale up the image dimension with linear activations
    d1 = Dense(input_shape[0] * input_shape[1])(em)
    # reshape to additional channel
    d1 = Reshape((input_shape[0], input_shape[1], 1))(d1)
    # image input
    image_input = Input(shape=input_shape, name="imagen_discriminador")
    #  concate label as channel
    merge = Concatenate()([image_input, d1])
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    fe = LeakyReLU(0.2)(fe)
    fe = Dropout(0.7)(fe)
    fe = BatchNormalization()(fe)

    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    fe = LeakyReLU(0.2)(fe)
    fe = Dropout(0.5)(fe)
    fe = BatchNormalization()(fe)

    # flatten feature maps
    fe = Flatten()(fe)

    # ouput
    out_layer = Dense(1, activation='sigmoid')(fe)
    # define model
    model = tf.keras.models.Model([image_input, in_labels], out_layer)
    # compile model
    return model


# define standalone generator model
def define_generator(latent_dim, n_classes=7):
    # label input
    label_input = Input(shape=(1,), name="label_generador")
    # Embedding layer
    em = Embedding(n_classes, 50)(label_input)
    nodes = 25 * 25

    em = Dense(nodes)(em)
    em = Reshape((25, 25, 1))(em)
    # image generator input
    image_input = Input(shape=(latent_dim,), name="ruido_espacio_latente")
    nodes = 128 * 25 * 25
    d1 = Dense(nodes)(image_input)
    d1 = LeakyReLU(0.2)(d1)
    d1 = Reshape((25, 25, 128))(d1)
    # merge
    merge = Concatenate()([d1, em])
    # upsample to 50x50
    gen = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(merge)
    gen = Dropout(0.7)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(0.2)(gen)

    # upsample to 100x100
    gen = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(gen)
    gen = Dropout(0.7)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(0.2)(gen)

    # upsample to 300x300
    gen = Conv2DTranspose(64, (3, 3), strides=(3, 3), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(0.2)(gen)

    # output layer
    out_layer = Conv2D(1, (7, 7), activation='tanh', padding='same')(gen)
    # define model
    model = tf.keras.models.Model([image_input, label_input], out_layer)
    return model


class cGAN(tf.keras.models.Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.discriminator = discriminator

    def compile(self, opt_g, opt_d, loss_g, loss_d, *args, **kwargs):
        super().compile(*args, **kwargs)

        self.opt_g = opt_g
        self.opt_d = opt_d
        self.loss_g = loss_g
        self.loss_d = loss_d

    @tf.function
    def train_step(self, batch):
        batch_size = tf.shape(batch[0])[0]
        latent_dim = self.generator.input_shape[0][1]

        images_real, labels = batch

        labels = tf.expand_dims(labels, axis=-1)

        # Generate images
        latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        images_generated = self.generator([latent_vectors, labels], training=False)

        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator([images_real, labels], training=True)
            yhat_fake = self.discriminator([images_generated, labels], training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Create labels for real and fakes images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            # Add some noise to the TRUE outputs (crucial step)
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # Calculate loss
            total_loss_d = self.loss_d(y_realfake, yhat_realfake)

        # Apply backpropagation
        dgrad = d_tape.gradient(total_loss_d, self.discriminator.trainable_variables)
        self.opt_d.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate images
            latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
            images_generated = self.generator([latent_vectors, labels], training=True)

            # Create the predicted labels
            predicted_labels = self.discriminator([images_generated, labels], training=False)

            # Calculate loss - trick to training to fake out the discriminator
            total_loss_g = self.loss_g(tf.zeros_like(predicted_labels), predicted_labels)

        # Apply backpropagation
        ggrad = g_tape.gradient(total_loss_g, self.generator.trainable_variables)
        self.opt_g.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"loss_d": total_loss_d, "loss_g": total_loss_g}


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    X, y, _ = dataloader.DataLoader(data_dir=f"../{constants.data.FINAL_PATH}/groundtruth.csv",
                                    aps_list=constants.aps, batch_size=30, step_size=5,
                                    size_reference_point_map=300,
                                    return_axis_coords=False)()
    minimo, maximo = np.min(X), np.max(X)
    X_reescalado = 2 * (X - minimo) / (maximo - minimo) - 1

    y_encoded = dataloader.labelEncoding(y)
    dataset = tf.data.Dataset.from_tensor_slices((X_reescalado, y_encoded)).shuffle(800).batch(8)

    os.makedirs(constants.outputs.models.training, exist_ok=True)
    os.makedirs(constants.outputs.models.cgan_300, exist_ok=True)

    path_images_training = f"../{constants.outputs.models.cgan_300}/images_training"
    path_checkpoints = f"../{constants.outputs.models.cgan_300}/checkpoints"
    path_learning_curves = f"../{constants.outputs.models.cgan_300}/learning_curves"

    os.makedirs(path_images_training, exist_ok=True)
    os.makedirs(path_checkpoints, exist_ok=True)
    os.makedirs(path_learning_curves, exist_ok=True)

    for latent_dim in [100, 300, 500, 700, 1000]:
        tf.keras.backend.clear_session()

        # Definición de los directorios de salida de las imágenes, los checkpoints y las curvas de aprendizaje
        out_images = f"{path_images_training}/latent_dim={latent_dim}"
        out_checkpoints = f"{path_checkpoints}/latent_dim={latent_dim}"
        out_learning_curves = f"{path_learning_curves}/latent_dim={latent_dim}"

        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_checkpoints, exist_ok=True)
        os.makedirs(out_learning_curves, exist_ok=True)

        # definición del modelo
        discriminator = define_discriminator()
        generator = define_generator(latent_dim)
        gan = cGAN(generator, discriminator)

        # definición de los callbacks
        save_image = SaveImageTraining(X_reescalado, y_encoded, save_dir=out_images)
        save_model = LoggingCheckpointTraining(save_dir=out_checkpoints)
        decay_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss_g', factor=0.5, patience=5, min_lr=0.00001)
        hist = tf.keras.callbacks.History()

        # Compilación y entrenamiento del modelo
        gan.compile(tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, clipnorm=1.0),
                    tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5),
                    tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.BinaryCrossentropy())
        gan.fit(dataset, epochs=50, callbacks=[hist, save_image, save_model, decay_lr])

        # Curvas de aprendizaje
        plt.plot(hist.history["loss_d"], label="loss_d")
        plt.plot(hist.history["loss_g"], label="loss_g")
        plt.title("Curvas de aprendizaje")
        plt.legend()
        plt.savefig(f"{out_learning_curves}/learning_curves.png")
        plt.close()
