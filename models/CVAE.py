import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import time
import os


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    def call(self, inputs):
        x = inputs
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decoder(z)
        return x_logit

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(tf.shape(inputs)[0], self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        batch_size = tf.shape(mean)[0]
        eps = tf.random.normal(shape=(batch_size, mean.shape[1]))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def compile(self, optimizer):
        super(CVAE, self).compile()
        self.optimizer = optimizer

    def generate_and_save_images(self, epoch, x_test, y_test, random_vector_for_generation, model_name="vae_mnist"):
        os.makedirs(model_name, exist_ok=True)
        os.makedirs(f"{model_name}/latent_dim", exist_ok=True)
        os.makedirs(f"{model_name}/generated_images", exist_ok=True)

        # Display a 2D plot of the digit classes in the latent space
        z_mean = self.encoder.predict(x_test, batch_size=128)
        if z_mean.shape[1]//2 == 2:
            plt.figure(figsize=(12, 10))
            plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
            plt.colorbar()
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.savefig(f"{model_name}/latent_dim/latent {epoch:04d}.png")
            plt.show()

        predictions = self.decode(random_vector_for_generation, apply_sigmoid=True)
        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.savefig(f"{model_name}/generated_images/ generated {epoch:04d}.png")
        plt.show()


if __name__ == "__main__":
    train_size = 60000
    batch_size = 128
    test_size = 10000
    optimizer = tf.keras.optimizers.Adam(1e-4)

    (train_images, _), (test_images, y_test) = tf.keras.datasets.mnist.load_data()

    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size))

    epochs = 200
    latent_dim = 2
    num_examples_to_generate = 16

    random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
    model = CVAE(latent_dim)
    model.compile(optimizer)
    model.fit(train_dataset, epochs=20)

    model.generate_and_save_images(20, x_test=test_images, y_test=y_test, random_vector_for_generation=random_vector_for_generation, model_name="vae_mnist")
