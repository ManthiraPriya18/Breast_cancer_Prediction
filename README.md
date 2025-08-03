# Breast_cancer_Prediction
# 🧠 Breast Cancer Prediction using Support Vector Machine (SVM)

This project uses the Breast Cancer Wisconsin dataset to build a predictive model using **Support Vector Machines (SVM)** for classifying tumors as benign or malignant. It includes preprocessing, model tuning with GridSearchCV, and evaluation.

## 🔍 Project Highlights

- ✅ Dataset: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- ✅ Preprocessing with `StandardScaler`
- ✅ Model: SVM (`sklearn.svm.SVC`)
- ✅ Hyperparameter tuning using `GridSearchCV`
- ✅ Evaluation using accuracy score on test data

---

## 📊 Dataset Overview

- **Features:** 30 real-valued input features (e.g., radius, texture, smoothness)
- **Classes:** 
  - 0 → malignant  
  - 1 → benign  
- **Samples:** 569 instances

---

## 🚀 How it Works

1. **Load and explore the dataset**
2. **Split into train (80%) and test (20%) sets**
3. **Standardize features**
4. **Tune hyperparameters using GridSearchCV**:
   - `C`: Regularization strength (e.g., 0.1, 1, 10)
   - `kernel`: Kernel type (e.g., linear, rbf, poly)
   - `gamma`: Kernel coefficient (e.g., scale, auto)
5. **Evaluate best model on the test set**

---

## 📈 Results

- ✅ **Best Parameters:** Retrieved using GridSearchCV
- ✅ **Cross-Validation Accuracy:** Reported during tuning
- ✅ **Test Accuracy:** Final evaluation on unseen data

---

## 📓 Notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ManthiraPriya18/Breast_cancer_Prediction/blob/main/Breast_Cancer_detection_1.ipynb)

---

## 🛠️ Requirements

- Python 3.x
- NumPy
- Scikit-learn
- Jupyter / Google Colab

Install dependencies (if running locally):

```bash
pip install numpy scikit-learn
