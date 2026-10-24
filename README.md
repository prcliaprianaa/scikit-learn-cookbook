# scikit-learn Cookbook

This repository contains notebooks with code, theoretical explanations, and chapter summaries based on the book **scikit-learn Cookbook: Over 80 recipes for machine learning in Python with scikit-learn** (3rd Edition, Packt Publishing, 2025), available on O'Reilly Learning.

This work is an individual assignment for the Machine Learning enrichment class.

## Repository Structure

```
scikit-learn-cookbook/
├── Chapter1_API_Elements_of_scikit-learn/
│   └── Chapter1_API_Elements_of_scikit-learn.ipynb
├── Chapter2_Data_Preprocessing/
│   └── Chapter2_Data_Preprocessing.ipynb
├── Chapter3_Dimensionality_Reduction/
│   └── Chapter3_Dimensionality_Reduction.ipynb
├── Chapter4_Distance_Metrics_and_Nearest_Neighbors/
│   └── Chapter4_Distance_Metrics_and_Nearest_Neighbors.ipynb
├── Chapter5_Linear_Models_and_Regularization/
│   └── Chapter5_Linear_Models_and_Regularization.ipynb
├── Chapter6_Advanced_Logistic_Regression/
│   └── Chapter6_Advanced_Logistic_Regression.ipynb
├── Chapter7_Support_Vector_Machines_and_Kernel_Methods/
│   └── Chapter7_Support_Vector_Machines_and_Kernel_Methods.ipynb
├── Chapter8_Tree-Based_Algorithms_and_Ensemble_Methods/
│   └── Chapter8_Tree-Based_Algorithms_and_Ensemble_Methods.ipynb
├── Chapter9_Text_Processing_and_Multiclass_Classification/
│   └── Chapter9_Text_Processing_and_Multiclass_Classification.ipynb
├── Chapter10_Clustering_Techniques/
│   └── Chapter10_Clustering_Techniques.ipynb
├── Chapter11_Novelty_and_Outlier_Detection/
│   └── Chapter11_Novelty_and_Outlier_Detection.ipynb
├── Chapter12_Cross-Validation_and_Model_Evaluation/
│   └── Chapter12_Cross-Validation_and_Model_Evaluation.ipynb
├── Chapter13_Deploying_Models_in_Production/
│   ├── Chapter13_Deploying_Models_in_Production.ipynb
│   ├── app.py                       # Example REST API (FastAPI)
│   └── models/                      # Saved pipelines and model metadata
└── README.md
```

Each notebook contains:
- Python code examples for the chapter's topics, with outputs
- Theoretical explanations of each concept, including the key formulas
- Interpretation of the results
- A summary of the chapter

Most examples use datasets that are built into scikit-learn (such as Iris, Wine, Digits, Breast Cancer, and Diabetes) or synthetic data generated in the notebook. Chapter 9 downloads the 20 Newsgroups dataset automatically through scikit-learn.

## Chapter Overview

| # | Chapter | Topics Covered |
|:--|:--|:--|
| 1 | API Elements of scikit-learn | Design philosophy, estimators, transformers, custom components, pipelines, hyperparameters, metadata |
| 2 | Data Preprocessing | Imputation, scaling, normalization, encoding, discretization, feature engineering, ColumnTransformer |
| 3 | Dimensionality Reduction | PCA, Kernel PCA, Truncated SVD, LDA, t-SNE, feature selection |
| 4 | Distance Metrics and Nearest Neighbors | Distance metrics, KNN classification and regression, KD-trees, Ball trees |
| 5 | Linear Models and Regularization | Linear regression, polynomial features, Ridge, Lasso, ElasticNet, SGD |
| 6 | Advanced Logistic Regression | Odds ratios, thresholds, ROC and AUC, regularization, imbalanced data, multiclass |
| 7 | Support Vector Machines and Kernel Methods | Maximum margin, soft margin, kernels, gamma, SVR, LinearSVC |
| 8 | Tree-Based Algorithms and Ensemble Methods | Decision trees, pruning, bagging, random forests, boosting, voting, stacking |
| 9 | Text Processing and Multiclass Classification | Bag-of-words, TF-IDF, n-grams, hashing, text pipelines, OvR, OvO, output codes |
| 10 | Clustering Techniques | K-Means, Mini-Batch K-Means, hierarchical clustering, DBSCAN, HDBSCAN, Gaussian mixtures |
| 11 | Novelty and Outlier Detection | z-score, IQR, Elliptic Envelope, Isolation Forest, LOF, One-Class SVM |
| 12 | Cross-Validation and Model Evaluation | K-Fold, stratified, group and time-series CV, metrics, hyperparameter search, nested CV, learning curves |
| 13 | Deploying Models in Production | Serialization, metadata, input validation, REST API, batch inference, drift monitoring, retraining |

### Chapter 1: Common Conventions and API Elements of scikit-learn
Introduces the design principles that make scikit-learn consistent and easy to use: consistency, simplicity, modularity, and reusability. Covers estimators and the `fit()`, `predict()`, and `fit_predict()` methods, transformers with `fit()`, `transform()`, and `fit_transform()`, and the train/test rule that prevents data leakage. It also shows how to build custom transformers and classifiers with `BaseEstimator` and mixins, chain steps into pipelines and save them with joblib, inspect learned attributes such as `coef_` and `intercept_`, evaluate models with `score()` and adjusted R-squared, tune hyperparameters with `get_params()`, `set_params()`, `GridSearchCV`, and `RandomizedSearchCV`, and use estimator tags and metadata routing.

### Chapter 2: Pre-Model Workflow and Data Preprocessing
Covers the steps that turn raw data into model-ready input. Topics include splitting data before preprocessing, handling missing values with `SimpleImputer`, `KNNImputer`, and `IterativeImputer`, scaling with `StandardScaler`, `MinMaxScaler`, `RobustScaler`, and `MaxAbsScaler`, per-sample normalization with `Normalizer`, encoding categories with `OneHotEncoder` and `OrdinalEncoder`, discretization and binarization, creating new features with `PolynomialFeatures` and `FunctionTransformer`, and combining everything into one leak-free workflow with `ColumnTransformer` and `Pipeline`.

### Chapter 3: Dimensionality Reduction Techniques
Explains how to reduce the number of features to speed up training, enable visualization, and reduce overfitting. Covers feature extraction with PCA (explained variance and choosing the number of components), Kernel PCA for nonlinear structure, Truncated SVD for sparse data, supervised Linear Discriminant Analysis, and t-SNE for visualization. It also covers feature selection with `VarianceThreshold`, `SelectKBest`, recursive feature elimination (RFE), and `SelectFromModel`.

### Chapter 4: Building Models with Distance Metrics and Nearest Neighbors
Covers distance metrics (Euclidean, Manhattan, Minkowski, Chebyshev, and cosine) and how they define similarity. Topics include K-Nearest Neighbors for classification and regression, the importance of feature scaling, choosing k and the weighting scheme with cross-validation, the Nearest Centroid classifier, efficient neighbor search with KD-trees and Ball trees, and unsupervised neighbor search with `NearestNeighbors`.

### Chapter 5: Linear Models and Regularization
Covers linear regression and how regularization prevents overfitting. Topics include ordinary least squares, overfitting with polynomial features, Ridge regression (L2 penalty) and coefficient paths, Lasso regression (L1 penalty) for automatic feature selection, ElasticNet, choosing the penalty strength with `RidgeCV`, `LassoCV`, and `ElasticNetCV`, and Stochastic Gradient Descent regression for large datasets.

### Chapter 6: Advanced Logistic Regression and Extensions
Goes beyond basic logistic regression. Topics include the sigmoid function and log loss, interpreting coefficients as odds ratios, adjusting the decision threshold, evaluating classifiers with the confusion matrix, precision, recall, F1-score, ROC curves, AUC, and precision-recall curves, L1 and L2 regularization with `LogisticRegressionCV`, handling imbalanced classes with `class_weight`, and multiclass classification with softmax and One-vs-Rest.

### Chapter 7: Support Vector Machines and Kernel Methods
Covers Support Vector Machines for classification and regression. Topics include the maximum margin and support vectors, the soft margin and the `C` parameter, the kernel trick with linear, polynomial, and RBF kernels, the effect of `gamma`, the need for feature scaling, tuning kernels and hyperparameters with `GridSearchCV`, Support Vector Regression with an epsilon-insensitive tube, and `LinearSVC` for larger datasets.

### Chapter 8: Tree-Based Algorithms and Ensemble Methods
Covers decision trees and ensembles built from them. Topics include Gini and entropy impurity, controlling tree depth and cost-complexity pruning, regression trees, bagging, random forests with out-of-bag scores, Extra Trees, impurity-based and permutation feature importance, boosting with AdaBoost, Gradient Boosting, and HistGradientBoosting, and combining different models with voting and stacking.

### Chapter 9: Text Processing and Multiclass Classification
Covers how to turn text into numeric features and classify documents into several categories. Topics include the bag-of-words model with `CountVectorizer`, stop words, n-grams, and vocabulary control, TF-IDF weighting, `HashingVectorizer` for large-scale text, text classification pipelines with Multinomial Naive Bayes, logistic regression, and linear SVMs on the 20 Newsgroups dataset, interpreting the most important words per class, evaluating multiclass models with macro, weighted, and micro averages, multiclass strategies (One-vs-Rest, One-vs-One, and error-correcting output codes), and tuning text pipelines with `GridSearchCV`.

### Chapter 10: Clustering Techniques
Covers unsupervised methods for grouping similar data points. Topics include K-Means with k-means++ initialization, choosing the number of clusters with the elbow method and silhouette analysis, Mini-Batch K-Means for large datasets, agglomerative clustering with different linkage methods and dendrograms, density-based clustering with DBSCAN (including the k-distance plot for choosing eps) and HDBSCAN, Gaussian Mixture Models for soft clustering with model selection by BIC, and evaluating clusters with internal metrics (silhouette, Calinski-Harabasz, Davies-Bouldin) and external metrics (ARI, NMI).

### Chapter 11: Novelty and Outlier Detection
Covers how to detect observations that differ strongly from the rest of the data. Topics include the difference between outlier detection and novelty detection, simple statistical rules (z-score and IQR), Elliptic Envelope with the Mahalanobis distance, Isolation Forest, Local Outlier Factor in outlier and novelty mode, One-Class SVM, the role of the `contamination` parameter and anomaly scores, evaluating detectors with precision, recall, F1, and ROC AUC, and comparing how the detectors behave on data of different shapes.

### Chapter 12: Cross-Validation and Model Evaluation Techniques
Covers how to estimate model performance reliably and tune models without overfitting to the evaluation. Topics include the limitations of a single holdout split, K-Fold, Stratified K-Fold, ShuffleSplit, and Leave-One-Out, group-aware and time-series splits, `cross_validate` and `cross_val_predict`, choosing metrics and creating custom scorers, dummy baselines, hyperparameter search with `GridSearchCV`, `RandomizedSearchCV`, and `HalvingGridSearchCV`, nested cross-validation, and diagnosing overfitting and underfitting with learning curves and validation curves.

### Chapter 13: Deploying scikit-learn Models in Production
Covers what happens after a model is trained. Topics include building a robust production pipeline that handles missing values and unknown categories, saving and loading models with joblib and pickle along with their security and version-compatibility risks, storing model metadata for versioning, wrapping the model in a prediction function with input validation, an example REST API with FastAPI, batch versus real-time inference and latency, monitoring for data drift with the Kolmogorov-Smirnov test and the Population Stability Index, and retraining within the MLOps model lifecycle.

## How to Run

1. Clone this repository:
```
git clone https://github.com/prcliaprianaa/scikit-learn-cookbook.git
cd scikit-learn-cookbook
```

2. Install the required libraries:
```
pip install scikit-learn numpy pandas scipy matplotlib joblib jupyter
```

3. Open any notebook in Jupyter or VS Code and run all cells.

4. (Optional) To try the example API from Chapter 13:
```
cd Chapter13_Deploying_Models_in_Production
pip install fastapi uvicorn
uvicorn app:app --reload
```
Then open http://127.0.0.1:8000/docs in a browser.

## Notes

- Chapter 1 of the book is mostly conceptual and has no official code notebook, so its notebook focuses on explanations with small illustrative examples.
- The notebooks follow the topics of each chapter in the book. Some examples use different datasets or were adapted to work with recent versions of scikit-learn.
- Chapter 9 needs an internet connection the first time it runs, to download the 20 Newsgroups dataset.
- Some notebooks create output files when they run (saved models in Chapter 1 and the `models/` folder and `app.py` in Chapter 13). These files are part of the examples.
- Theoretical explanations were written with the assistance of an LLM, as permitted by the assignment, and reviewed by the author.

## References

- *scikit-learn Cookbook: Over 80 recipes for machine learning in Python with scikit-learn*, 3rd Edition. Packt Publishing, 2025.
- Official code repository: https://github.com/PacktPublishing/Scikit-Learn-Cookbook-Third-Edition
- scikit-learn documentation: https://scikit-learn.org/stable/