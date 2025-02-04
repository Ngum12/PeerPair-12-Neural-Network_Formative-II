# Neural Network Classifier with Advanced Optimization Techniques

## Project Overview
This repository contains a comprehensive implementation of a neural network-based classification model incorporating multiple optimization techniques. Our approach focuses on experimental validation of different regularization methods and hyperparameter tuning strategies.

## 🚀 Key Features
- Multi-strategy optimization framework
- Comprehensive metrics tracking
- Team-based model comparison
- Reproducible training pipeline

## 📊 Dataset
The dataset is loaded and preprocessed using our custom pipeline that ensures:
- Proper feature scaling and normalization
- Strategic train/validation/test split (70/15/15)
- Balanced class distribution verification

## 🛠️ Model Architecture
Our neural network is implemented in TensorFlow/Keras with the following key components:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Define the optimized Adam model
def create_adam_model():
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    # Compile the model with Adam optimizer and a lower learning rate
    optimizer = ()
    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Instantiate the model
model = create_model()

```

## 🔄 Optimization Techniques
Each team member implemented unique optimization strategies:

| Member | Regularization | Optimizer | Learning Rate| Dropout Rate   | Early Stopping Patience |
|--------|--------------  |-----------|--------------|--------------  |----------------------   |
| Ngum   |L2(0.001)       | Adam      | 0.0005       | 0.4 → 0.3 → 0.2|          15             |
|  Marion|L2(0.001,0.01)  | SGD       |   0.0005     |  0.4           |       4                 |
| Aubin  |L2(0.005)       | RMSProp   | 0.0004       |  0.2          |      10                    |

## 📈 Performance Metrics

### Model Comparison Results

| Model  | Accuracy | F1 Score  | Precision  | Recall   |
|------- |----------|---------- |----------- |--------- |
| Ngum   |0.70      | 0.78/0.52 | 0.72/0.62  | 0.86/0.44|
| Marion |0.65      | 0.71/0.55 | 0.70/0.57  | 0.71/0.54|
| Aubin  |0.68      | 0.78/0.43 | 0.69/0.64  | 0.89/0.33|

[The video to our presentation](https://drive.google.com/drive/folders/1QuyV-atXAYeEz5cR0BEYwNM8ZHgwicW2?usp=sharing)


Accuracy:
Ngum and Aubin had the highest accuracy rate compared to Marion: 70%, 68%, and 65% respectively. This suggests that Marion's model may have had more difficulty in generalizing across the entire dataset.

F1 Score:
Comparing everyone's F1 score, it is evident that all three models struggled with class imbalance, with higher scores for class 0 and lower scores for class 1. However, Marion's model seemed to have a balanced score compared to the rest.

Precision & Recall:
The three models showed that it was easy to identify class 0 but had a harder time distinguishing class 1. 


## 🔍 Analysis Insights

### Performance Analysis
- Ngum's model achieved superior performance due to:
  1. Balanced dropout rate (0.4) preventing overfitting while maintaining model capacity
  2. Conservative learning rate (0.0005) enabling stable convergence
  3. Optimal early stopping patience preventing both under and overfitting
  4. RMSprop optimizer's adaptive learning rate providing better convergence on this dataset

- Marion's model performed like this:
  1. It achieved a train accuracy of 0.762 but dropped to 0.649 on the test accuracy. This shows that the model may have somewhat overfitted the training data despite my use of dropout and early stopping.
  2. The model performs better at predicting class 0 (with a precision of 0.70 and recall of 0.71) but struggles with class 1 (precision of 0.57 and recall of 0.54). This imbalance may be due to class distribution in the data.
  3. The F1 score for class 0 (0.71) is much better than for class 1 (0.55), reflecting a higher balance between precision and recall for class 0.
  4. The choice of a small learning rate (0.0005) with SGD seems to be effective in the data training.
 
- Aubin's model shows this learning dynamics:
  1. The training accuracy steadily increases to around 0.71 while validation accuracy plateaus at approximately 0.68, indicating relatively good generalization with minimal overfitting despite the growing gap.
  2. The loss curves show rapid initial convergence in the first 10 epochs, followed by a stable plateau around 0.6 for both training and validation loss, suggesting the model found a stable optimization point.
  3. The validation accuracy shows periodic fluctuations between 0.67-0.68 throughout training, while maintaining overall stability, which could indicate the model is sensitive to certain batches in the validation set.
  4. The consistent convergence pattern suggests the RMSprop optimizer with learning rate 0.0004 provided effective gradient updates.

### Key Findings
- Ngum:
  1. Higher dropout rates (>0.5) led to underfitting.
  2. L1 regularization showed better feature selection compared to L2
  3. Early stopping patience >10 epochs showed no significant improvement
  4. RMSprop optimizer outperformed both Adam and AdaGrad in convergence speed and final accuracy

- Marion:
  1. The SGD optimizer with a low learning rate and dropout contributed to a stable training process, but it also led to a notable gap between train and test accuracy.
  2. Precision and recall metrics show a stronger performance on the majority class (class 0), indicating the need to address class imbalance further.
  3. Exploring different regularization techniques and optimizing the dropout rate and early stopping criteria could help balance the performance across classes, especially improving recall for class 1.

- Aubin
  1. The model's stronger performance on Class 1 (recall 0.89) versus Class 2 (recall 0.33) suggests potential class imbalance issues that could be addressed through sampling techniques or class weights.
  2. The combination of L2 regularization (0.005) and moderate dropout (0.2) appears effective at maintaining a reasonable gap between training and validation performance, though there's room for improvement in closing this gap further.
     
## 🚦 Getting Started

### Prerequisites
```bash
python>=3.8
tensorflow>=2.5.0
numpy>=1.19.2
pandas>=1.2.4
scikit-learn>=0.24.2
```

### Installation
```bash
git clone https://github.com/your-repo/neural-network-classifier.git
cd neural-network-classifier
pip install -r requirements.txt
```

### Running the Models
```bash
python train.py --member alice --regularizer l2 --optimizer adam --learning_rate 0.001
```

## 📁 Repository Structure
```
neural-network-classifier/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── alice_model/
│   ├── bob_model/
│   └── carol_model/
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── tests/
├── requirements.txt
└── README.md
```

## 👥 Team Contributions
- **Ngum**: Data preprocessing, L2 regularization implementation, Adam optimizer tuning
- **Marion**: Model architecture design, L1 regularization studies, RMSprop implementation
- **Aubin**: Performance analysis, L1L2 regularization experiments, AdaGrad optimization

## 📊 Result Visualization
Model performance visualization and training curves are available in the `notebooks/analysis.ipynb` file.

##  Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
