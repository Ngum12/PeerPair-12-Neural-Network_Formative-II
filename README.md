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
def create_model(input_dim, learning_rate, dropout_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.Precision(), 
                         tf.keras.metrics.Recall()])
    return model
```

## 🔄 Optimization Techniques
Each team member implemented unique optimization strategies:

| Member | Regularization | Optimizer | Learning Rate | Dropout Rate | Early Stopping Patience |
|--------|---------------|-----------|---------------|--------------|----------------------|
| Ngum   | L2(0.01)      | Adam      | 0.001         | 0.3          | 5 epochs             |
| Marion |L2(0.001, 0.01)| SGD       | 0.0005        | 0.4          | 4 epochs             |
|        | L1L2(0.01)    | AdaGrad   | 0.0001        | 0.4          | 10 epochs            |

## 📈 Performance Metrics

### Model Comparison Results

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|-----------|-----------|---------|
| Marion| 0.76/0.64| 0.71/0.55 | 0.70/0.57 |0.71/0.54|
| Bob   | 0.91     | 0.90      | 0.92      | 0.89    |
| Carol | 0.88     | 0.87      | 0.86      | 0.88    |

## 🔍 Analysis Insights

### Performance Analysis
- Marion's model performed like this:
  1. It achieved a train accuracy of 0.762 but dropped to 0.649 on the test accuracy. This shows that the model may have somewhat overfitted the training data despite my use of dropout and early stopping.
  2. The model performs better at predicting class 0 (with a precision of 0.70 and recall of 0.71) but struggles with class 1 (precision of 0.57 and recall of 0.54). This imbalance may be due to class distribution in the data.
  3. The F1 score for class 0 (0.71) is much better than for class 1 (0.55), reflecting a higher balance between precision and recall for class 0.
  4. The choice of a small learning rate (0.0005) with SGD seems to be effective in the data training.
     
### Key Findings
1. The SGD optimizer with a low learning rate and dropout contributed to a stable training process, but it also led to a notable gap between train and test accuracy.
2. Precision and recall metrics show a stronger performance on the majority class (class 0), indicating the need to address class imbalance further.
3. Exploring different regularization techniques and optimizing the dropout rate and early stopping criteria could help balance the performance across classes, especially improving recall for class 1.

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
- **Alice**: Data preprocessing, L2 regularization implementation, Adam optimizer tuning
- **Bob**: Model architecture design, L1 regularization studies, RMSprop implementation
- **Carol**: Performance analysis, L1L2 regularization experiments, AdaGrad optimization

## 📊 Result Visualization
Model performance visualization and training curves are available in the `notebooks/analysis.ipynb` file.

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
