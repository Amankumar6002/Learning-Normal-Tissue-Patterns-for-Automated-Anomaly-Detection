import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# PROJECT: Learning Normal Tissue Patterns for Anomaly Detection using VAE
# This script defines the VAE architecture used for medical image reconstruction.
# -----------------------------------------------------------------------------

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(input_shape=(256, 256, 1), latent_dim=128):
    # 1. ENCODER: Compress image to latent space
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # 2. DECODER: Reconstruct image from latent space
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64 * 64 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((64, 64, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # 3. VAE MODEL: Connect Encoder and Decoder
    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
                )
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss + kl_loss
            
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

    return VAE(encoder, decoder)

# -----------------------------------------------------------------------------
# ANOMALY DETECTION LOGIC
# -----------------------------------------------------------------------------
def detect_anomalies(model, image):
    """
    Anomalies are detected by comparing the original image with its VAE reconstruction.
    Since the VAE is trained only on 'Normal' tissue, it will fail to reconstruct
    'Abnormal' features, creating a high reconstruction error (anomaly).
    """
    # 1. Get Reconstruction
    _, _, z = model.encoder.predict(image)
    reconstructed_img = model.decoder.predict(z)
    
    # 2. Calculate Residual (Difference)
    residual = np.abs(image - reconstructed_img)
    
    # 3. Generate Heatmap
    anomaly_heatmap = np.sum(residual, axis=-1)[0]
    return reconstructed_img[0], anomaly_heatmap

# --- Example Usage ---
if __name__ == "__main__":
    # Simulate loading 100 normal medical scans (256x256 grayscale)
    x_train = np.random.rand(100, 256, 256, 1).astype("float32")
    
    # Initialize and Train VAE
    vae = build_vae()
    vae.compile(optimizer=keras.optimizers.Adam())
    print("Starting training on Normal Tissue patterns...")
    vae.fit(x_train, epochs=5, batch_size=16)
    
    # Test Anomaly Detection on a new image
    test_img = np.random.rand(1, 256, 256, 1).astype("float32")
    reconstructed, heatmap = detect_anomalies(vae, test_img)
    
    print("Analysis complete. Heatmap generated.")
    # vae.save('medical_vae_model') # Save for later conversion to model.json
