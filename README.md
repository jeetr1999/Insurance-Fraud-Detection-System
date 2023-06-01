# Insurance-Fraud-Detection-System

## Overview:
In this project, we aim to develop a fraud detection system using three different models: Random Forest Classifier, Isolation Forest, and a Neural Network. The dataset contains information related to fraudulent activities, and the goal is to accurately predict whether a transaction is fraudulent or not. We will evaluate the performance of each model and compare their accuracies.

## Methods:

### 1. Random Forest Classifier:
   - We import the required libraries, including RandomForestClassifier from the sklearn.ensemble module and accuracy_score from the sklearn.metrics module.
   - A Random Forest Classifier model is created with 100 estimators (decision trees) and a random state of 42.
   - The model is trained on the training data (X_train and y_train).
   - Predictions are made on the test data (X_test) using the trained model.
   - The accuracy of the model is calculated using the accuracy_score function by comparing the predicted labels (y_pred) with the actual labels (y_test).
   - The accuracy is printed on the console.

### 2. Isolation Forest:
   - We import the required libraries, including train_test_split from the sklearn.model_selection module, IsolationForest from the sklearn.ensemble module, and StandardScaler from the sklearn.preprocessing module.
   - The features (X) and target variable (y) are separated from the dataset.
   - The data is split into training and test sets using train_test_split with a test size of 0.2 and a random state of 42.
   - The features are standardized separately for the train and test sets using StandardScaler.
   - An Isolation Forest model is created.
   - The model is trained on the standardized training data (X_train_scaled).
   - Predictions are made on the standardized test data (X_test_scaled) using the trained model.
   - The accuracy of the model is not explicitly mentioned in the provided code.

### 3. Neural Network:
   - We import the required libraries, including tf.keras from TensorFlow and set the random seed to 42 using tf.random.set_seed(42).
   - A Sequential model is created using tf.keras.Sequential().
   - The model architecture includes multiple dense layers with different activation functions (e.g., 'relu') and varying numbers of neurons.
   - The model is compiled using tf.keras.losses.mae as the loss function, tf.keras.optimizers.Adam(lr=0.1) as the optimizer with a learning rate of 0.1, and 'accuracy' as the metric for evaluation.
   - The model is fit to the standardized training data (X_train_scaled and y_train) for 50 epochs.
   - The training history, including loss and accuracy values, is stored in the 'history' variable.

## Results:
-	The accuracy for Neural Network Model and Random Forest Regression was 93.7%.

## Conclusion:
In addition to the Random Forest Classifier and Isolation Forest models, a Neural Network model using a Sequential architecture has been implemented for fraud detection. The model consists of multiple dense layers and is trained using the standardized training data. The accuracy achieved by the Neural Network model is not explicitly mentioned in the code snippet. Evaluating and comparing the accuracies of all three models would provide a comprehensive understanding of their performance and assist in selecting the most effective model for fraud detection.
