# Marketing Campaign Response Prediction Project Overview
* The main objective of this project is to train a predictive model that predicts who will respond to an offer for a product or service. as it allows the company to maximize the profit of the next marketing campaign.
* Analyzed the data of a fictitious company to select approachable customers and increase the profit of a marketing campagin.
* Developed a response model that provides a significant boost to the efficiency of a marketing campaign by increasing responses or reducing expenses. 

## Code and Resources
**Python Version:** 3.8   
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, scipy     
**Modules:** MinMaxScaler, PowerTransformer, OneHotEncoder, PCA, KBinsDiscretizer, LogisticRegression, GaussianNB, RandomForestClassifier, Pipeline, GridSearchCV, ColumnTransformer  
**Dataset:** Kaggle - targeted_marketing_campaign   
**Reference:** https://medium.com/ai-in-plain-english/targeted-marketing-with-machine-learning-38de28162483

## Categorical Features Exploratory Analysis
<p float="left">
  <img src="https://github.com/Wei-Chong-Eden/Marketing_Campaign/blob/master/images/Discrimination_ability_of_categories.png" width="600" />
</p>

## Numerical Features Exploratory Analysis
<p float="left">
  <img src="https://github.com/Wei-Chong-Eden/Marketing_Campaign/blob/master/images/Numerical_Features_Correlation_Matrix.png" width="600" />
</p>
<p float="left">
  <img src="https://github.com/Wei-Chong-Eden/Marketing_Campaign/blob/master/images/Discribution_by_response.png" width="600" />
</p>

## Train-Test-Split
Split data before further processing to avoid data leakage that makes model evaluation overly optimistic.

## Multivariate Outlier Detection and Removal with Mahalanobis Distance
Mahalanobis distance is an effective multivariate distance metric that measures the distance between a point and a distribution. It is an extremely useful metric having, excellent applications in multivariate anomaly detection, classification on highly imbalanced datasets and one-class classification. 

### Feature Engineering
Create new business-oriented features:
* PrpGoldProds: Proportion of Monetary Units spent on gold product out of the total spent
* PrpWines: Proportion of Monetary Units spent on wines out of the total spent
* PrpFruits: Proportion of Monetary Units spent on fruits out of the total spent
* PrpMeatProds: Proportion of Monetary Units spent on meat product out of the total spent
* PrpFishProds: Proportion of Monetary Units spent on fish product out of the total spent
* PrpSweetProds: Proportion of Monetary Units spent on sweet product out of the total spent
* MntTotal: Monetary (Total spend)
* Frequency
* BuyPot: Buy Potential (proportion of total spent and income)
* NmbAccCmps: Number of Accepted Campaigns out of the last 5 Campaigns
* RFM

### Data Transformation
* Feature Scaling on numerical attributes: MinMaxScaler
* Power Transformation on numerical attributes: A power transform will make the probability distribution of a variable more Gaussian when the data is skewed. The transformed training dataset can then be fed to the classification model.
* Convert categorical attributes: A one hot encoding allows the representation of categorical data to be more expressive

### Transformation Pipeline
Automate the whole pre-processing process (from feature engineering to data transfomation)

### Feature Extration & Selection
* Use PCA to figure out the most imporant features explained variance by each component
* Chi-Square Test for Categorical Variables and Binned Continuous Variables
  <p float="left">
  <img src="https://github.com/Wei-Chong-Eden/Marketing_Campaign/blob/master/images/Features'_worth_by_Chi-Squared_statistic_test.png" width="600" />
  </p>
* Random Forest and Feature Importances

## Model Evaluation using Cross-Validation
* Split the training set into a smaller training set and a validation set, then train the models against the smaller training set and evaluate them against the validation set. 
* Compare various models using ROC curves and ROC AUC scores.
<p float="left">
  <img src="https://github.com/Wei-Chong-Eden/Marketing_Campaign/blob/master/images/Classifier_ROC_Curve_and%20ROC_AUC_scores_Comparison_Cross-Validation.png" width="600" />
</p>

## Evaluate the System on the Test Set
Evaluation models on test set using ROC curves and ROC AUC scores.    
<p float="left">
  <img src="https://github.com/Wei-Chong-Eden/Marketing_Campaign/blob/master/images/Classifier_evaluation_on_test_set.png" width="600" />
</p>
