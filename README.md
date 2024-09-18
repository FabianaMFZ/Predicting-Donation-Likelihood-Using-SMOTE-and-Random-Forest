## Predicting-Donation-Likelihood-Using-SMOTE-and-Random-Forest

### OVERVIEW

Tackled an imbalanced classification problem using a dataset of customer donations. The goal was to predict whether a customer would make a donation (Target_B), using Random Forests and SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes.
- **Problem:**

The target variable TARGET_B was highly imbalanced, with significantly more customers who didn't make donations. Using this data directly would result in a biased model. The objective was to create a model that could accurately predict the likelihood of a donation while addressing class imbalance.

- **Data Preparation:**

Conducted Exploratory Data Analysis (EDA) to investigate correlations and feature relationships.

Performed T-tests to filter out features with significant differences between donation and non-donation groups, improving feature selection.

Encoded categorical variables using OneHotEncoding to prepare the dataset for modeling.

- **Handling Class Imbalance:**

Applied SMOTE to balance the dataset by upscaling the minority class, which resulted in equal representation of donation and non-donation records in the training data.

Used Logistic Regression to identify significant predictors of donations and improve model interpretability.

- **Modeling:**

Tried different Decision Tree depths to evaluate model performance and plotted the Recall score as the evaluation metric.

Tuned a Random Forest Classifier using grid search with different combinations of n_estimators and max_depth.

The best model used 100 estimators and a maximum depth of 31, achieving a strong AUC (Area Under the Curve) score during validation.


### RESULTS
- The Random Forest model achieved excellent performance, effectively handling the imbalanced dataset with SMOTE.
- The model demonstrated high recall and AUC scores, indicating strong predictive power for identifying customers likely to donate.

![image](https://github.com/user-attachments/assets/9891d1e2-385c-42d5-a4ec-4f1d1f80fc79)




