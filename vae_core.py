import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses
import matplotlib.pyplot as plt

"""
MEDICAL AI PROJECT: VAE ANOMALY DETECTION
Core Python Implementation for Model Training and Inference
This script defines the Variational Autoencoder architecture used to learn 
normal tissue patterns for automated anomaly detection.
"""

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def get_encoder(input_shape, latent_dim):
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def get_decoder(latent_dim, output_shape):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(output_shape[0] // 4 * output_shape[1] // 4 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((output_shape[0] // 4, output_shape[1] // 4, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    return models.Model(latent_inputs, decoder_outputs, name="decoder")

class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"loss": self.total_loss_tracker.result()}

def run_training_demo():
    # Configuration
    IMG_SIZE = 128
    LATENT_DIM = 32
    EPOCHS = 10
    
    # Generate Mock Medical Data (Normal Scans)
    print("Generating synthetic 'Normal' medical data for training...")
    x_train = np.random.normal(0.5, 0.1, (100, IMG_SIZE, IMG_SIZE, 1))
    x_train = np.clip(x_train, 0, 1).astype("float32")

    # Build Model
    encoder = get_encoder((IMG_SIZE, IMG_SIZE, 1), LATENT_DIM)
    decoder = get_decoder(LATENT_DIM, (IMG_SIZE, IMG_SIZE, 1))
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam())

    # Training
    print(f"Starting training for {EPOCHS} epochs...")
    vae.fit(x_train, epochs=EPOCHS, batch_size=16, verbose=1)

    # Inference & Anomaly Detection Simulation
    test_img = x_train[0:1]
    # Add a fake anomaly (bright spot)
    test_img[0, 40:60, 40:60, 0] = 1.0 
    
    _, _, z = vae.encoder.predict(test_img)
    reconstruction = vae.decoder.predict(z)
    
    # Calculate Anomaly Map (Residual)
    anomaly_map = np.abs(test_img[0] - reconstruction[0])
    
    print("Inference complete. Anomaly map generated.")
    return test_img[0], reconstruction[0], anomaly_map

if __name__ == "__main__":
    orig, recon, diff = run_training_demo()
    print("Process Finished. Ready for model export to TensorFlow.js.")
