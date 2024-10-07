# Nowcasting Convective Rainfall in West Africa
## Overview
This project focuses on building a ConvLSTM model for nowcasting convective rainfall in the West African region. The goal is to predict short-term rainfall intensity based on satellite imagery, which is crucial for timely weather forecasting, especially in regions prone to rapid storm formation and where ground-based weather infrastructure is limited.

## Aim
The main objective is to accurately predict rainfall intensity at short intervals (up to 2 hours) using ConvLSTM, leveraging the spatiotemporal dependencies in satellite data to enhance real-time forecasting.

## Methodology
### Model Architecture:
__Input__: Sequential satellite imagery frames representing rainfall intensity over time.

__Layers__:
- ConvLSTM layers to capture spatiotemporal features.
- Conv3D layers to refine and generate predictions for the next rainfall frame.

__Output__: Predicted rainfall intensity for future frames.

__Loss Function__: Mean Squared Error (MSE) - This loss function was chosen to minimize the squared difference between the predicted and actual rainfall intensities, making it ideal for continuous prediction tasks.

__Evaluation Metrics__:
- Fractional Skill Score (FSS): Measures spatial accuracy across various window sizes (3x3, 5x5, 10x10).
- Accuracy & AUC: Overall performance metrics assessed using accuracy and Area Under the Curve (AUC) via ROC curves.

__Data__:
Satellite-derived rainfall imagery specific to West Africa’s mesoscale convective systems is used as input data for nowcasting.

__Computational Resources__:
The ARC4 high-performance computing cluster was used to train the ConvLSTM models, offering the computational power necessary for large-scale data processing and deep learning model training.

__Model Configurations__:
Four different ConvLSTM architectures were designed and tested, each with varying input and output configurations:

- Model 1:
Input: Frames 1, 2, 3.
Output: Frames 4 and 5.
- Model 2:
Input: Frames 1, 2, 3, 4.
Output: Frame 5.
- Model 3:
Input: Frames 1 and 3.
Output: Frames 4 and 5.
- Model 4:
Input: Frames 1 and 4.
Output: Frame 5.

## Results
Average Accuracy: The model achieved a test accuracy of 97.3%.

FSS Scores: The highest FSS score was 0.68 for Model 2 (window size 3x3).

AUC: High AUC values demonstrate the model’s robust performance in distinguishing convective rainfall events.

## Conclusion
The ConvLSTM models developed in this project outperform traditional NWP (Numerical Weather Prediction) models, improving short-term rainfall prediction accuracy by 1.5%. These models are highly effective for real-time weather forecasting, offering crucial predictions for regions with complex weather systems like West Africa.

## File Structure
1. model1model1.py, model2model2.py, model3model3.py, model4model4.py:
These files contain the model architectures for the four ConvLSTM models developed in the project. Each script defines the layers, activation functions, and optimizers used to train the respective models. The architectures differ slightly to explore different configurations for predicting rainfall intensity.
The models take sequential satellite images as input and predict future rainfall intensities based on the spatiotemporal patterns captured by ConvLSTM layers.

2. train_val_loss.py:
This script visualizes the training and validation loss curves for each of the models. It is useful for monitoring overfitting and model performance during training.

3. accuracy_plot.py:
This file generates the accuracy plots for all four models, comparing their performance across epochs. This is essential for evaluating the consistency of model predictions and how well they generalize to unseen data.

4. fss.py:
These scripts are responsible for calculating and visualizing the Fractional Skill Score (FSS), which evaluates the spatial accuracy of the rainfall predictions at different window sizes. The FSS is a key metric in assessing the model’s ability to capture the spatial distribution of rainfall.

5. prediction1.py, prediction2.py, prediction3.py, prediction4.py:
These scripts are used to generate the prediction plots comparing the true and predicted rainfall maps for each model. Each script corresponds to one of the four models and outputs visual comparisons that are useful for analyzing model performance.

6. roc1.py, roc2.py, roc3.py, roc4.py:
These scripts generate the Receiver Operating Characteristic (ROC) curves and calculate the Area Under the Curve (AUC) for each model. This allows for a thorough evaluation of the models' ability to differentiate between different classes (rainfall/no rainfall) and is particularly helpful when dealing with binary classification tasks in weather prediction.

## Future Work
- Enhancements: Explore the integration of additional weather data (e.g., wind patterns, humidity).
- Advanced Architectures: Investigate hybrid architectures combining ConvLSTM with attention mechanisms for enhanced performance.
- Deployment: Extend the current model for real-time nowcasting applications in West Africa.
