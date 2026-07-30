from __future__ import annotations

import tensorflow as tf


def build_denoising_autoencoder(spectrum_length: int) -> tf.keras.Model:
    """Build a compact 1D-CNN autoencoder suitable for quantization."""
    inputs = tf.keras.Input(shape=(spectrum_length, 1), name="noisy_spectrum")
    x = tf.keras.layers.Conv1D(12, 7, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(8, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(4, 3, padding="same", activation="relu", name="spectral_bottleneck")(x)
    x = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(12, 5, padding="same", activation="relu")(x)
    outputs = tf.keras.layers.Conv1D(1, 3, padding="same", name="denoised_spectrum")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="pcf_spr_denoiser")


def build_ri_predictor(spectrum_length: int) -> tf.keras.Model:
    """Predict analyte RI and resonance wavelength from a denoised spectrum."""
    inputs = tf.keras.Input(shape=(spectrum_length, 1), name="spectrum")
    x = tf.keras.layers.Conv1D(12, 7, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(20, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(24, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, name="ri_lambda_res")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="pcf_spr_ri_predictor")
