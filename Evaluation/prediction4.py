import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('model4model4_lr.h5')

# Load the test dataset
test_dataset = np.load('test_dataset.npy')

# Prepare the test data
def create_shifted_frames(data):
    x = data[:, [0, 3], :, :, :]  # Use frames 1 and 4 as input
    y = data[:, 4, :, :, :]       # Frame 5 as output
    return x, y

x_test, y_test = create_shifted_frames(test_dataset)

# Make predictions on the test dataset
predictions = model.predict(x_test)

# Plot and save each sample with its predictions
n_samples = 5  # Number of samples to plot

for i in range(n_samples):
    plt.figure(figsize=(15, 5))

    # Plot true frame 1
    plt.subplot(1, 3, 1)
    plt.imshow(x_test[i, 0, :, :, 0], cmap='magma')
    plt.title(f'True Frame 1 (Sample #{i+1})')
    plt.axis('off')

    # Plot true frame 4
    plt.subplot(1, 3, 2)
    plt.imshow(x_test[i, 1, :, :, 0], cmap='magma')
    plt.title(f'True Frame 4 (Sample #{i+1})')
    plt.axis('off')

    # Plot predicted frame 5
    plt.subplot(1, 3, 3)
    plt.imshow(predictions[i, :, :, 0], cmap='magma')
    plt.title(f'Predicted Frame 5 (Sample #{i+1})')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'model4_{i+1}.png')
    plt.show()

print("Plotting and saving complete.")

