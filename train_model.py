import numpy as np
import tensorflow as tf
from keras import layers, Model
from keras import backend as K
# === Load data ===
X1 = np.load("X1.npy")
X2 = np.load("X2.npy")
y = np.load("y.npy")

# === Clean check ===
assert set(np.unique(y)) == {0, 1}, "y ต้องมีแค่ 0 กับ 1"
assert not np.isnan(X1).any(), "X1 มี NaN"
assert not np.isnan(X2).any(), "X2 มี NaN"
assert not np.isinf(X1).any(), "X1 มี inf"
assert not np.isinf(X2).any(), "X2 มี inf"

# === Normalize ===
X1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)

# === Siamese base model ===
def build_base_model(input_shape=(99,)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    return Model(inputs, x)
base_model = build_base_model()

# === Siamese input/output ===
input_1 = layers.Input(shape=(99,))
input_2 = layers.Input(shape=(99,))

emb_1 = base_model(input_1)
emb_2 = base_model(input_2)

# === L2 distance (safe version) ===
@tf.keras.utils.register_keras_serializable()
class L2Distance(tf.keras.layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True) + 1e-6)

distance = L2Distance(name="l2_distance")([emb_1, emb_2])


# === Siamese Model ===
siamese_model = Model(inputs=[input_1, input_2], outputs=distance)
@tf.keras.utils.register_keras_serializable()
# === Contrastive Loss ===
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    return tf.reduce_mean(
        y_true * tf.square(y_pred) +
        (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    )

@tf.keras.utils.register_keras_serializable()
# === Siamese Accuracy ===
def siamese_accuracy(y_true, y_pred):
    threshold = 0.3  # กำหนด threshold สำหรับการตัดสินใจ
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.cast(y_true, tf.bool), tf.less_equal(y_pred, threshold)), tf.float32)
    )

# === Compile ===
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
siamese_model.compile(optimizer=optimizer, loss=contrastive_loss, metrics=[siamese_accuracy])
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# === Train ===
if __name__ == "__main__":
    history = siamese_model.fit(
        [X1, X2], y,
        batch_size=32,
        epochs=150,
        validation_split=0.2,
    )
    siamese_model.save("siamese_model.keras")
# === Save ===
siamese_model.save("siamese_model.keras")  # ✅ รูปแบบใหม่
converter = tf.lite.TFLiteConverter.from_keras_model(siamese_model)
tflite_model = converter.convert()
with open("siamese_model.tflite", "wb") as f:
    f.write(tflite_model)
    
print("X1 max/min:", X1.max(), X1.min())
print("X2 max/min:", X2.max(), X2.min())
print("Any NaN in X1?", np.isnan(X1).any())
print("Any NaN in X2?", np.isnan(X2).any())
print("Any Inf in X1?", np.isinf(X1).any())
print("Any Inf in X2?", np.isinf(X2).any())

# print("✅ บันทึก siamese_model_contrastive_fixed.h5 แล้ว")
# print("X1 shape:", X1.shape)
# print("X2 shape:", X2.shape)
# print("y shape:", y.shape)
# print("y unique:", np.unique(y))
# print("X1 max/min:", X1.max(), X1.min())
# print("NaN in X1:", np.isnan(X1).any())
# print("Inf in X1:", np.isinf(X1).any())
# unique, counts = np.unique(y, return_counts=True)
# print(dict(zip(unique, counts)))  # ต้องใกล้เคียงกัน