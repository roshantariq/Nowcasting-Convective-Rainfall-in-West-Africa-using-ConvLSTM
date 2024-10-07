import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def calc_fss_window(data_true, data_pred, threshold, window_size):
    score = 0.0
    total_windows = 0

    for i in range(data_true.shape[1] - window_size + 1):
        for j in range(data_true.shape[2] - window_size + 1):
            window_true = data_true[:, i:i+window_size, j:j+window_size, :]
            window_pred = data_pred[:, i:i+window_size, j:j+window_size, :]
            
            num = np.sum((window_true >= threshold) & (window_pred >= threshold))
            denom = np.sum((window_true >= threshold) | (window_pred >= threshold))
            
            score += num / denom if denom != 0 else 0
            total_windows += 1

    return score / total_windows if total_windows > 0 else 0

# Load test data
test_dataset = np.load('test_dataset.npy')  # make sure to adjust the path as needed

# Assuming the test_dataset is already divided into features and labels
x_test, y_test = test_dataset[:, :4, :, :, :], test_dataset[:, 4:, :, :, :]

models = ["model1model1_lr.h5", "model2model2_lr.h5", "model3model3_lr.h5", "model4model4_lr.h5"]  # Adjust model paths as necessary
window_sizes = [3, 5, 10]
threshold = 0.5

fss_scores = {}

for model_name in models:
    model = load_model(model_name)
    predictions = model.predict(x_test)

    fss_scores[model_name] = []
    for window_size in window_sizes:
        fss = calc_fss_window(y_test, predictions, threshold, window_size)
        fss_scores[model_name].append(fss)
        print(f'Model {model_name}, Window Size {window_size}, FSS: {fss:.4f}')

# Optionally, save or print the FSS scores in a more structured form
import pandas as pd

fss_data = []
for model, scores in fss_scores.items():
    for window_size, score in zip(window_sizes, scores):
        fss_data.append({
            'Model': model,
            'Window Size': window_size,
            'FSS': score
        })

df = pd.DataFrame(fss_data)
print(df)
# Optionally, save to CSV
df.to_csv('fss_scores.csv', index=False)

