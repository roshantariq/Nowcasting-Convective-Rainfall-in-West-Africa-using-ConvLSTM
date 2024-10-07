import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, callbacks
import json

# Ensure GPU is used
print("Setting up GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f'{len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)')
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")

# Load and normalize data
def load_and_normalize_data(file_name):
    data = np.load(file_name)
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Prepare data for training
def create_shifted_frames(data):
    x = data[:, :4, :, :, :]  # First 4 frames as input
    y = data[:, 4:, :, :, :]  # Next 1 frames as output
    return x, y

# Load datasets
print("Loading datasets...")
train_dataset = np.load('train_dataset.npy')
val_dataset = np.load('val_dataset.npy')
test_dataset = np.load('test_dataset.npy')

# Prepare data for training
x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)
x_test, y_test = create_shifted_frames(test_dataset)

# Print dataset shapes
print(f"Train dataset shape: {x_train.shape}, {y_train.shape}")
print(f"Validation dataset shape: {x_val.shape}, {y_val.shape}")
print(f"Test dataset shape: {x_test.shape}, {y_test.shape}")

# Define the model architecture
def build_model(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.ConvLSTM2D(128, (5, 5), padding='same', return_sequences=True, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ConvLSTM2D(32, (3, 3), padding='same',return_sequences=False, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = models.Model(inputs=inp, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['accuracy'])
    return model

model = build_model((4, 256, 256, 1))
model.summary()

# Custom callback to print losses and accuracy
class PrintLossCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'End of epoch {epoch + 1}: Training loss: {logs["loss"]:.6f}, Validation loss: {logs["val_loss"]:.6f}, Training accuracy: {logs["accuracy"]:.6f}, Validation accuracy: {logs["val_accuracy"]:.6f}')

# Train the model
print("Starting model training...")
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=8,
    callbacks=[PrintLossCallback(), callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)]
)

# Save the trained model and training history
model.save('model2model2_lr.h5')
print("Trained model saved.")

# Convert float32 in history to float for JSON serialization
def convert_history_to_float(history):
    return {k: [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v] for k, v in history.items()}

history_dict = convert_history_to_float(history.history)

# Ensure the history object is fully written to the JSON file
try:
    with open('model2model2_lr.json', 'w', encoding='utf-8') as f:
        json.dump(history_dict, f, ensure_ascii=False, indent=4)
    print("History successfully saved to model1model1_history.json")
except Exception as e:
    print(f"Error saving history: {e}")

# Validate that the JSON file was saved correctly
try:
    with open('model2model2_lr.json', 'r', encoding='utf-8') as f:
        history_check = json.load(f)
    print("History JSON file is valid and fully written.")
except json.JSONDecodeError as e:
    print(f"Error in saved JSON file: {e}")

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

print("Script completed successfully.")

