import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('model2model2_lr.h5')

# Load the test dataset
test_dataset = np.load('test_dataset.npy')

# Prepare the test data
def create_shifted_frames(data):
    x = data[:, :4, :, :, :]  # first 4 frames as input
    y = data[:, 4:, :, :, :]  # frame 5 as output
    return x, y

x_test, y_test = create_shifted_frames(test_dataset)

# Make predictions on the test dataset
predictions = model.predict(x_test)

# Plot and save each sample with its predictions
n_samples = 5  # Number of samples to plot

for i in range(n_samples):
    plt.figure(figsize=(18, 5))
    
    # Plot all five true frames
    for j in range(5):
        plt.subplot(1, 6, j + 1)
        plt.imshow(test_dataset[i, j, :, :, 0], cmap='magma')
        plt.title(f'True Frame {j+1} (Sample #{i+1})')
        plt.axis('off')
    
    # Plot the predicted frame (Frame 5)
    plt.subplot(1, 6, 6)
    plt.imshow(predictions[i, :, :, 0], cmap='magma')
    plt.title(f'Predicted Frame 5 (Sample #{i+1})')
    plt.axis('off')
    
    # Save the plot for the current sample
    plt.tight_layout()
    plt.savefig(f'model2lr_{i+1}.png')
    plt.close()

print("Plotting and saving complete.")

