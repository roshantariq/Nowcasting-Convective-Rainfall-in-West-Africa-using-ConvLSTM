import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc

# Function to create shifted frames for input/output data preparation
def create_shifted_frames(data):
    X = data[:, [0,3], :, :, :]  # Assume first 4 frames as input if the model uses 4 frames
    y = data[:, 4, :, :,:]  # Assume the 5th frame as output, adapt if necessary
    return X, np.expand_dims(y, axis=1)  # Ensuring y has the correct shape

def load_data(file_path):
    # Load the dataset
    data = np.load(file_path)
    return create_shifted_frames(data)

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Model 4')
    plt.legend(loc="lower right")
    plt.show()

# Load model
model = load_model('model4model4_lr.h5')

# Load test dataset and prepare it using create_shifted_frames
X_test, y_test = load_data('test_dataset.npy')

# Predictions
predictions = model.predict(X_test).flatten()  # Flatten predictions to 1D

# Set threshold as the mean of the predictions to create a binary prediction
threshold = np.mean(predictions)
binary_predictions = (predictions >= threshold).astype(int)
binary_labels = (y_test.flatten() >= threshold).astype(int)  # Flatten y_test to match prediction shape

# Calculate FPR, TPR, and AUC
fpr, tpr, _ = roc_curve(binary_labels, predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plot_roc_curve(fpr, tpr, roc_auc)

