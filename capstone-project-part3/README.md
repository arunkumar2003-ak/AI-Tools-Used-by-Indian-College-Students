# Part 3 – Advanced Modeling, Ensembles, Hyperparameter Tuning and ML Pipeline

## Overview

In this part of the project, multiple machine learning classification models were trained and compared to identify the most robust model. Ensemble learning techniques, cross-validation, hyperparameter tuning, feature importance analysis, feature ablation, pipeline creation, and model serialization were implemented using Scikit-learn.

---

# Dataset

* Dataset Used: **Cleaned Dataset** (generated in Part 1)
* Target Variable:

  * Regression Target: `Impact_on_Grades`
  * Classification Target:

```python
y_clf = (Impact_on_Grades > median).astype(int)
```

The classification target converts the regression target into two classes based on the median value.

---

# Task 1 – Default Decision Tree

A default `DecisionTreeClassifier` was trained without restricting the depth of the tree.

### Results

* Training Accuracy: **1.0000**
* Testing Accuracy: **0.7676**

### Interpretation

The training accuracy reached 100%, while the testing accuracy was much lower. This indicates **overfitting**.

Decision Trees are considered **high-variance models** because they greedily split the training data at each node to minimize impurity. Small changes in the training data can produce a completely different tree.

---

# Task 2 – Controlled Decision Tree

Model Parameters:

* max_depth = 5
* min_samples_split = 20

### Results

* Training Accuracy: **0.6427**
* Testing Accuracy: **0.5989**

### Parameter Explanation

### max_depth

Limits how deep the tree can grow.

A smaller tree reduces overfitting but may increase bias.

### min_samples_split

Prevents splitting nodes containing fewer than 20 samples.

This avoids creating branches based on very small noisy subsets.

### Comparison

Compared to the default tree, the constrained tree has a much smaller train-test gap. In this dataset it also underfits slightly, giving lower overall accuracy.

---

# Task 3 – Gini vs Entropy

### Formula

Gini Impurity

```
Gini = 1 − Σ(pi²)
```

Entropy

```
Entropy = −Σ pi log₂(pi)
```

A node with **Gini = 0** contains samples from only one class.

### Results

* Gini Test Accuracy: **0.5989**
* Entropy Test Accuracy: **0.6279**

Entropy performed slightly better than Gini on this dataset.

---

# Task 4 – Random Forest

Model Parameters

* n_estimators = 100
* max_depth = 10
* random_state = 42

### Results

* Training Accuracy: **0.6264**
* Testing Accuracy: **0.6058**
* ROC-AUC: **0.7008**

## Top 5 Important Features

| Feature                                     | Importance |
| ------------------------------------------- | ---------: |
| Daily_Usage_Hours                           |   0.050628 |
| Awareness_Level                             |   0.034127 |
| College_Name_Indian Institute of Technology |   0.030537 |
| Do_Professors_Allow_Use_Yes                 |   0.029431 |
| Trust_in_AI_Tools                           |   0.028280 |

### Feature Importance

Random Forest computes feature importance using the **average reduction in Gini impurity** produced by splits on each feature across all trees.

Unlike linear regression coefficients, feature importance values measure how useful a feature is for reducing prediction error rather than the direction or magnitude of a linear relationship.

---

# Bagging

Random Forest is an ensemble learning algorithm based on **Bagging (Bootstrap Aggregating)**.

Each tree is trained using a bootstrap sample (random sampling with replacement) from the training data.

At every split, only a random subset of features is considered.

The predictions from all trees are averaged (or voted) to reduce variance and improve generalization compared to a single Decision Tree.

---

# Task 4a – Gradient Boosting

Model Parameters

* n_estimators = 100
* learning_rate = 0.1
* max_depth = 3

### Results

* Training Accuracy: **0.6973**
* Testing Accuracy: **0.6279**
* ROC-AUC: **0.6706**

---

# Task 4b – Feature Ablation

The five least important features were removed and the Random Forest was retrained.

### Results

| Model         |    ROC-AUC |
| ------------- | ---------: |
| Full Model    | **0.7008** |
| Reduced Model | **0.6995** |

### Interpretation

The ROC-AUC changed only slightly after removing the five least important features. This indicates that these features contributed very little to model performance. Removing such features can simplify deployment by reducing inference time and maintenance while preserving predictive performance.

---

# Task 5 – Cross Validation

Five-fold Stratified Cross Validation was performed using ROC-AUC.

| Model               | Mean AUC | Std AUC |
| ------------------- | -------: | ------: |
| Logistic Regression |   0.7507 |  0.0170 |
| Decision Tree       |   0.6195 |  0.0120 |
| Random Forest       |   0.7036 |  0.0077 |
| Gradient Boosting   |   0.6761 |  0.0107 |

### Why Cross Validation?

Cross-validation evaluates the model on multiple train-validation splits, providing a more reliable estimate of generalization performance than a single train-test split.

---

# Task 6 – GridSearchCV

Parameter Grid

* n_estimators = [50, 100, 200]
* max_depth = [5, 10, None]
* min_samples_leaf = [1, 5]

Total Configurations

```
3 × 3 × 2 = 18
```

Total Fits

```
18 × 5 = 90
```

### Best Parameters

* max_depth = None
* min_samples_leaf = 1
* n_estimators = 200

### Best Cross Validation ROC-AUC

**0.8571**

### Grid Search vs Random Search

Grid Search evaluates every possible parameter combination and can find the optimal configuration but is computationally expensive. Randomized Search evaluates only a subset of parameter combinations, making it faster while often achieving similar performance.

---

# Task 7 – Manual Learning Curve

The tuned pipeline was trained using 20%, 40%, 60%, 80%, and 100% of the training data.

Training and testing ROC-AUC values were compared to study model behavior.

### Conclusion

As more training data becomes available, the training performance typically decreases slightly while the testing performance becomes more stable. This helps determine whether the model is limited by data quantity or by model capacity.

---

# Task 8 – Model Serialization

The best model was saved using:

```python
joblib.dump(best_pipeline, "best_model.pkl")
```

The saved model was successfully reloaded using:

```python
joblib.load("best_model.pkl")
```

Predictions on sample test records executed successfully.

---

# Final Model Comparison

| Model               | CV Mean AUC | CV Std AUC | Test ROC-AUC |
| ------------------- | ----------: | ---------: | -----------: |
| Logistic Regression |      0.7507 |     0.0170 |            - |
| Decision Tree       |      0.6195 |     0.0120 |            - |
| Random Forest       |      0.7036 |     0.0077 |       0.7008 |
| Gradient Boosting   |      0.6761 |     0.0107 |       0.6706 |

---

# Final Recommendation

Based on the experimental results, Logistic Regression achieved the highest mean cross-validation ROC-AUC, indicating the best overall generalization performance on this dataset. Random Forest provided useful feature importance analysis and competitive ROC-AUC, making it valuable for model interpretability. Gradient Boosting performed reasonably well but did not outperform Logistic Regression. Therefore, Logistic Regression is recommended for deployment because it provides the strongest cross-validation performance while maintaining a simple and efficient model.
