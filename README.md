# 🌲 Forest Cover Type Classification with Deep Learning

## Project Overview

In this project, I developed deep learning models to predict **forest cover type** (the most common tree cover) for a given 30×30 meter land area using **cartographic and environmental features**. The dataset originates from the US Forest Service (USFS) Region 2 Resource Information System and contains information about elevation, slope, distances to natural features, wilderness areas, and soil types.

The task is a **multi-class classification problem** with **7 forest cover types**, ranging from very common to extremely rare classes. The covertypes are the following:

- Spruce/Fir
- Lodgepole Pine
- Ponderosa Pine
- Cottonwood/Willow
- Aspen
- Douglas-fir
- Krummholz

---

## Dataset Description

- **Rows:** 581,012  
- **Features:** 54 input features  
- **Target:** `class` (forest cover type, values 1–7)  
- **Missing values:** None  

### Feature types
- Continuous numerical features (e.g. elevation, slope, distances)
- Binary features (wilderness area indicators, soil types)

The dataset is **highly imbalanced**, with some forest cover types having far fewer examples than others.

---

## Data Preprocessing

The following preprocessing steps were applied:

1. **Target transformation**
   - Class labels were converted from `1–7` to `0–6` to meet TensorFlow/Keras requirements.

2. **Train / Validation / Test split**
   - 70% training  
   - 15% validation  
   - 15% test  
   - Stratified splitting was used to preserve class proportions.

3. **Feature scaling**
   - Numerical features were standardized using `StandardScaler`.
   - Scaling was fit on the training data only to avoid data leakage.

4. **Data storage**
   - Processed datasets were saved as NumPy arrays (`.npy`) for efficient reuse during training.

---

## Model Architectures

### Model v1 – Baseline Neural Network

- Fully connected (MLP) architecture  
- Two hidden layers:
  - Dense(256, ReLU)
  - Dense(128, ReLU)
- Output layer:
  - Dense(7, Softmax)
- Optimizer: Adam  
- Loss: Sparse Categorical Crossentropy  
- Early stopping based on validation loss  

**Purpose:**  
Establish a strong baseline and measure overall performance without explicitly addressing class imbalance.

---

### Model v2 – Class-Weighted Neural Network

- Same base architecture as Model v1  
- Added:
  - Class weights to penalize mistakes on rare classes
  - Dropout layer (0.3) for regularization  

**Purpose:**  
Improve recall for underrepresented forest cover types and study the trade-off between accuracy and class fairness.

---

## Evaluation Metrics

Because the dataset is imbalanced, multiple metrics were used:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Macro average (treats all classes equally)  
- Weighted average (accounts for class frequency)  
- Confusion matrix  

---

## Results Summary

### Model v1 (Baseline)

- **Test Accuracy:** ~90%  
- **Weighted F1-score:** ~0.90  
- **Macro F1-score:** ~0.85  

**Observations:**
- Excellent performance on majority classes
- Weaker recall for rare forest types
- High overall accuracy but biased toward frequent classes

---

### Model v2 (Class-Weighted)

- **Test Accuracy:** ~81%  
- **Macro Recall:** ~0.89 (significant improvement)
- Rare classes achieved **very high recall**, in some cases above 95%

**Observations:**
- Substantial improvement in detecting rare forest cover types
- Increased false positives for minority classes
- Lower overall accuracy due to the model prioritizing recall over precision

---

## Model Comparison and Discussion

The two models represent a **classic machine learning trade-off**:

- **Model v1** is best when overall accuracy and performance on common classes are the priority.
- **Model v2** is better when detecting rare forest cover types is more important, even at the cost of false positives.

In real-world ecological or environmental applications, missing rare forest types can be more costly than incorrectly flagging them, making Model v2 scientifically defensible despite its lower accuracy.

---

## Conclusion

This project demonstrates how deep learning models behave on large, real-world, imbalanced datasets. By comparing a baseline model with a class-weighted model, I showed how model design choices directly impact fairness, accuracy, and class-level performance.

The results highlight the importance of:
- Looking beyond accuracy
- Using appropriate evaluation metrics
- Aligning model selection with domain priorities

---

## Future Work

To further improve this project, future work could include:

- **Model v3:**  
  Train a new model using class weights **without dropout** or with reduced weight strength to balance recall improvements and overall accuracy.
- Hyperparameter tuning (learning rate, batch size, layer sizes)
- Using batch normalization
- Trying alternative algorithms (e.g. gradient boosting for tabular data)
- Feature importance analysis
- Deployment as a prediction API

---

## Tools & Libraries Used

- Python  
- NumPy  
- pandas  
- scikit-learn  
- TensorFlow / Keras  
- Matplotlib / Seaborn  
