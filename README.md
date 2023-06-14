# customerchurn

#%%Reading the data
#Reading the Data

import pandas as pd
import numpy as np

data1 = pd.read_excel(r'C:\Users\surbh\Downloads\Customer Churn Data.xlsx', sheet_name = 'Data for DSBA')
print(data1.head(10))
metadata = pd.read_excel(r'C:\Users\surbh\Downloads\Customer Churn Data.xlsx', sheet_name = 'Meta Data')
print(metadata)
#%%Data preprocessing

data1.drop('AccountID', axis = 1, inplace = True)
stats = data1.describe()
null_vals = data1.isnull().describe()
print(data1.dtypes)
print(data1.nunique())
data1['Tenure'] = data1['Tenure'].replace('#', np.nan)
data1['Account_user_count'] = data1['Account_user_count'].replace('@', np.nan)
data1['rev_per_month'] = data1['rev_per_month'].replace('+', np.nan)
data1['rev_growth_yoy'] = data1['rev_growth_yoy'].replace('$', np.nan)
data1['coupon_used_for_payment'] = data1['coupon_used_for_payment'].replace(['$','#','*'], np.nan)
data1['Day_Since_CC_connect'] = data1['Day_Since_CC_connect'].replace('$', np.nan)
data1['cashback'] = data1['cashback'].replace('$', np.nan)
data1['Login_device'] = data1['Login_device'].replace('&&&&', np.nan)
for values in data1['account_segment']:
    if values == 'Regular +':
        values == 'Regular Plus'
    elif values == 'Super +':
        values == 'Super Plus'
data1.dropna(axis = 0, inplace = True)
data1.to_excel('C:/Users/surbh/OneDrive/Desktop/Clean Data.xlsx')
#%%EDA
import matplotlib.pyplot as plt
import seaborn as sns

#Univariate Analysis

data1.hist(figsize=(15,15))
plt.show()

#Bivariate Analysis

sns.pairplot(data1)
plt.show()

#Correlation Analysis

corr = data1.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True)
plt.show()
#%%Decision Tree Classifier

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data1.iloc[:,4:5] = le.fit_transform(data1.iloc[:,4:5])
data1.iloc[:,5:6] = le.fit_transform(data1.iloc[:,5:6])
data1.iloc[:,8:9] = le.fit_transform(data1.iloc[:,8:9])
data1.iloc[:,10:11] = le.fit_transform(data1.iloc[:,10:11])
data1.iloc[:,-1:] = le.fit_transform(data1.iloc[:,-1:])

x = data1.iloc[:,1:]
y = data1.iloc[:,:1] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=(1))

decision_tree_clf = DecisionTreeClassifier(random_state = 1)
decision_tree_clf = decision_tree_clf.fit(x_train, y_train)

y_pred = decision_tree_clf.predict(x_test)

d_tree_acc = metrics.accuracy_score(y_test, y_pred)*100
print("Decision Tree Accuracy: ",d_tree_acc)
#%%Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr = lr.fit(x_train, y_train)

y_lr_pred = lr.predict(x_test)

lr_acc = metrics.accuracy_score(y_test, y_lr_pred)*100
print("Logistic Regression Accuracy: ", lr_acc)
#%%Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()
parameters = {'max_depth' : np.arange(1,10)}
grid = GridSearchCV(estimator = rf, param_grid = parameters)
grid.fit(x_train, y_train)
grid_best = grid.best_estimator_
print(grid_best)

rf = RandomForestClassifier(max_depth=9)
rf = rf.fit(x_train, y_train)

y_rf_pred = rf.predict(x_test)

rf_acc = metrics.accuracy_score(y_test, y_rf_pred)*100
print("Random Forest Classifier Accuracy: ", rf_acc)
#%%XgBoost
#import xgboost as xgb

#xg = xgb.XGBClassifier()
#xg_params = {'max_depth': np.arange(1,10), 'learning_rate': np.arange(0.01, 0.05)}
#grid_xgb = GridSearchCV(estimator = xg, param_grid = xg_params)
#grid_xgb.fit(x_train, y_train)

#grid_xg_best = grid_xgb.best_estimator_
#print(grid_xg_best)

#xg = xgb.XGBClassifier(max_depth = 9, learning_rate = 0.01, n_jobs = 8)
#xg.fit(x_train, y_train)

#y_xg_pred = xg.predict(x_test)

#xg_acc = metrics.accuracy_score(y_test, y_xg_pred)*100
#print("XgBoost Classifier Accuracy: ", xg_acc)
#%%Colated Results
result_dict = {'Models' : ['Decision Tree Classifier', 'Logistic Regression', 'Random Forest Classifier'], 
               'Accuracy Score' : [d_tree_acc, lr_acc, rf_acc]}
results_df = pd.DataFrame(result_dict)
print(results_df)
