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
|        | L1(0.001)     | RMSprop   | 0.0005        | 0.5          | 8 epochs             |
| Aubin  | L2(0.005)     | RMSprop   | 0.0004        | 0.2          | 10 epochs            |

## 📈 Performance Metrics

### Model Comparison Results

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|-----------|-----------|---------|
| Aubin  | 0.68     | 0.78/0.43| 0.69/0.64 | 0.89/0.33|
| Bob    | 0.91     | 0.90      | 0.92      | 0.89    |
| Carol  | 0.88     | 0.87      | 0.86      | 0.88    |

## 🔍 Analysis Insights

### Performance Analysis
Aubin's model shows interesting learning dynamics:

- The training accuracy steadily increases to around 0.71 while validation accuracy plateaus at approximately 0.68, indicating relatively good generalization with minimal overfitting despite the growing gap.
- The loss curves show rapid initial convergence in the first 10 epochs, followed by a stable plateau around 0.6 for both training and validation loss, suggesting the model found a stable optimization point.
- The validation accuracy shows periodic fluctuations between 0.67-0.68 throughout training, while maintaining overall stability, which could indicate the model is sensitive to certain batches in the validation set.
- The consistent convergence pattern suggests the RMSprop optimizer with learning rate 0.0004 provided effective gradient updates.

### Key Findings
- The combination of L2 regularization (0.005) and moderate dropout (0.2) appears effective at maintaining a reasonable gap between training and validation performance, though there's room for improvement in closing this gap further.
- The model's stronger performance on Class 1 (recall 0.89) versus Class 2 (recall 0.33) suggests potential class imbalance issues that could be addressed through sampling techniques or class weights.
- The training dynamics show good stability but potential for better generalization - adjusting the early stopping patience (currently 10 epochs) could help capture better model states given the oscillating validation accuracy pattern.

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
