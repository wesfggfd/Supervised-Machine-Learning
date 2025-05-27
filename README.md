# Decision Tree Algorithm Implementation Documentation

## Table of Contents
1. [Algorithm Overview](#algorithm-overview)  
2. [Loss Function Details](#loss-function-details)  
3. [Optimizers & Pruning Strategies](#optimizers--pruning-strategies)  
4. [Implementation & Parameters](#implementation--parameters)  
5. [Evaluation Metrics](#evaluation-metrics)  
6. [References](#references)  

---

## Algorithm Overview
Decision trees are supervised learning algorithms using tree structures for **classification** and **regression**. Core logic involves recursively partitioning the feature space to build predictive models[1,5](@ref).

### Algorithm Types
| Type              | Algorithms       | Characteristics                                                                 |
|-------------------|------------------|---------------------------------------------------------------------------------|
| **Classification** | ID3/C4.5/CART    | Split criteria: Information Gain, Gain Ratio, Gini Index                        |
| **Regression**      | CART             | Minimizes squared error for splits; leaf nodes output continuous value averages |

---

## Loss Function Details
### Classification Loss Functions
1. **Gini Index**  
   - Formula: $Gini(D) = 1 - \sum_{k=1}^K p_k^2$  
   - Advantages: Computationally efficient for CART[2,7](@ref)

2. **Information Entropy**  
   - Formula: $H(D) = -\sum_{k=1}^K p_k \log_2 p_k$  
   - Usage: Core metric for ID3/C4.5 algorithms[4,8](@ref)

3. **Gain Ratio**  
   - Addresses bias toward multi-valued features:  
     $GainRatio(D,A) = \frac{InfoGain(D,A)}{IV(A)}$  
     (IV = Intrinsic Value of feature A)[4](@ref)

### Regression Loss Functions
1. **Mean Squared Error (MSE)**  
   - Formula: $MSE = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$  
   - Default for CART regression trees[3](@ref)

2. **Mean Absolute Error (MAE)**  
   - Formula: $MAE = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$  
   - Robust to outliers[5](@ref)

---

## Optimizers & Pruning Strategies
### Optimization Methods
1. **Hyperparameter Tuning**  
   - Key parameters: `max_depth`, `min_samples_split`, `min_impurity_decrease`  
   - Methods: GridSearchCV/RandomizedSearchCV[6](@ref)

2. **Feature Selection**  
   - Recursive Feature Elimination (RFE) for complexity control[8](@ref)

### Pruning Techniques
| Type       | Implementation                                                                 | Pros/Cons                     |
|------------|--------------------------------------------------------------------------------|-------------------------------|
| Pre-pruning| Early stopping via conditions (e.g., max depth, min samples)                  | Fast but risks underfitting   |
| Post-pruning| Build full tree first, then prune using cost-complexity (CCP)                  | Better accuracy, computationally heavy[2,7](@ref) |

---

## Implementation & Parameters
### Classification Tree (Scikit-learn)


```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
   criterion='gini',        # 'gini' or 'entropy'
   max_depth=5,
   min_samples_split=10,
   ccp_alpha=0.01           # Cost-complexity pruning
)

clf.fit(X_train, y_train)
```



### Regression Tree (Scikit-learn)

 

```python
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(
criterion='mse',         # 'mse', 'friedman_mse', or 'mae'
max_depth=3,
min_samples_leaf=5
)

reg.fit(X_train, y_train)
```


---

## Evaluation Metrics
### Classification
| Metric       | Formula/Description                                  | Usage Scenario         |
|--------------|------------------------------------------------------|------------------------|
| Accuracy     | $\frac{TP+TN}{TP+TN+FP+FN}$                         | Balanced classes       |
| F1-Score     | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | Imbalanced data |

### Regression
| Metric       | Formula                                  | Characteristics         |
|--------------|------------------------------------------|-------------------------|
| R² Score     | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | Explains variance ratio |
| MAE          | ![](https://latex.codecogs.com/png.image?%5Cinline%20%5Cdpi%7B110%7D%5Cfrac%7B1%7D%7BN%7D%5Csum%7Cy_i-%5Chat%7By%7D_i%7C)     | Outlier-resistant       |

---

## References
[1](@ref): Decision Tree Fundamentals (Baidu Encyclopedia)  
[2](@ref): ID3/C4.5/CART Comparison (CSDN Blog)  
[3](@ref): CART Regression Implementation (CSDN Blog)  
[4](@ref): Information Gain vs. Gini Index (CDA Documentation)  
[5](@ref): Entropy Calculation Examples (Technical Blog)  
[6](@ref): Hyperparameter Optimization Guide (CSDN Library)  
[7](@ref): Practical Case Study (Blog Garden)  
[8](@ref): Decision Tree Stability Analysis (Research Paper)
