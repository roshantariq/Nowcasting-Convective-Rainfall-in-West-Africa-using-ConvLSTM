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
model_training.py: Code to train ConvLSTM models using satellite data.
data_preprocessing.py: Preprocessing pipeline, including data cleaning, normalization, and transformation.
evaluation_metrics.py: Functions to calculate FSS, AUC, and other evaluation metrics.
plot_predictions.py: Scripts to visualize the true rainfall maps versus model predictions.

## Future Work
Enhancements: Explore the integration of additional weather data (e.g., wind patterns, humidity).
Advanced Architectures: Investigate hybrid architectures combining ConvLSTM with attention mechanisms for enhanced performance.
Deployment: Extend the current model for real-time nowcasting applications in West Africa.
