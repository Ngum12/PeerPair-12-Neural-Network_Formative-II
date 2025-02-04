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
| Ngum   | L2(0.001)     | Adam      | 0.0005       | 0.4 → 0.3 → 0.2|          15         |
|        | L1(0.001)     | RMSprop   |               |              |                      |
|        | L1L2(0.01)    | AdaGrad   |               |             |                       |

## 📈 Performance Metrics

### Model Comparison Results

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|-----------|-----------|---------|
| Ngum   | 0.70    | 0.78-0.52| 0.72-0.62| 0.86-0.44|
| Bob    |          |           |        |       |
| Carol  |          |           |        |      |

## 🔍 Analysis Insights

### Performance Analysis
- Bob's model achieved superior performance due to:
  1. Balanced dropout rate (0.5) preventing overfitting while maintaining model capacity
  2. Conservative learning rate (0.0005) enabling stable convergence
  3. Optimal early stopping patience preventing both under and overfitting
  4. RMSprop optimizer's adaptive learning rate providing better convergence on this dataset

### Key Findings
1. Higher dropout rates (>0.5) led to underfitting
2. L1 regularization showed better feature selection compared to L2
3. Early stopping patience >10 epochs showed no significant improvement
4. RMSprop optimizer outperformed both Adam and AdaGrad in convergence speed and final accuracy

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
