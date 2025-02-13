# Deep Learning Model Comparison for Classification

## 📌 Overview
This project explores and compares multiple deep learning architectures for a classification task using different ensemble learning techniques:

- **Extra Trees Classifier**
- **Gradient Boosting Model**
- **Random Forest Regressor**

The deep learning models evaluated include:

- **CNN (Convolutional Neural Network)**
- **CNN-GRU (CNN + Gated Recurrent Units)**
- **CNN-LSTM (CNN + Long Short-Term Memory)**
- **DNN (Deep Neural Network)**

The goal is to analyze and compare their performance based on various evaluation metrics.

## 🔥 Features
- ✅ Implementation of multiple deep learning models
- ✅ Comparison of performance metrics (Accuracy, Precision, Recall, F1-Score)
- ✅ Evaluation using multiple ensemble learning techniques
- ✅ Dataset preprocessing & feature engineering

## 📊 Model Comparison Results

### **1️⃣ Extra Trees Classifier**
| Model       | Test Acc | Train Acc | Precision | Recall | F1 Score |
|------------|---------|----------|-----------|--------|----------|
| CNN        | 91.1%   | 92.4%    | 92.0%     | 91.0%  | 91.0%    |
| CNN-GRU    | 87.0%   | 89.6%    | 88.0%     | 87.0%  | 87.0%    |
| CNN-LSTM   | 93.5%   | 95.5%    | 94.0%     | 94.0%  | 93.0%    |
| DNN        | 88.3%   | 89.8%    | 89.0%     | 88.0%  | 88.0%    |

### **2️⃣ Gradient Boosting Model**
| Model       | Test Acc | Train Acc | Precision | Recall | F1 Score |
|------------|---------|----------|-----------|--------|----------|
| CNN        | 89.6%   | 91.5%    | 90.0%     | 90.0%  | 90.0%    |
| CNN-GRU    | 85.8%   | 86.0%    | 87.0%     | 86.0%  | 86.0%    |
| CNN-LSTM   | 90.8%   | 91.5%    | 91.0%     | 91.0%  | 91.0%    |
| DNN        | 88.9%   | 90.2%    | 89.0%     | 89.0%  | 89.0%    |

### **3️⃣ Random Forest Regressor**
| Model       | Test Acc | Train Acc | Precision | Recall | F1 Score |
|------------|---------|----------|-----------|--------|----------|
| CNN        | 89.2%   | 90.6%    | 90.0%     | 89.0%  | 89.0%    |
| CNN-GRU    | 90.9%   | 91.9%    | 91.0%     | 91.0%  | 91.0%    |
| CNN-LSTM   | 91.0%   | 92.3%    | 91.0%     | 91.0%  | 91.0%    |
| DNN        | 91.1%   | 91.8%    | 92.0%     | 91.0%  | 91.0%    |

## 🛠️ Tech Stack
- **Deep Learning Framework:** TensorFlow / PyTorch
- **Programming Language:** Python
- **Visualization:** Matplotlib, Seaborn
- **Data Processing:** Pandas, NumPy
- **Ensemble Learning Models:** Scikit-learn


## 📌 Key Insights
- **CNN-LSTM performed the best** across all models, achieving the highest test accuracy of **93.5%** with Extra Trees Classifier.
- **CNN models** generally outperformed DNN models due to their strong feature extraction capabilities.
- **Random Forest Regressor and Extra Trees Classifier** performed better than Gradient Boosting for most cases.
- **GRU-based models (CNN-GRU, DNN-GRU)** had the lowest accuracy compared to other architectures.

## 📌 Future Enhancements
- 🔹 Experiment with Transformer-based models.
- 🔹 Implement more advanced hyperparameter tuning.
- 🔹 Optimize models for real-time applications.

