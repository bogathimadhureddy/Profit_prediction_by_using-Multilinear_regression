import os 
os.chdir(r'C:\Users\madhu\OneDrive\Desktop\360 DigiTMG\DataScience\ASSIGNMENTS SOLVED BY ME\Multi Linear Regression')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
import sweetviz

from sklearn.model_selection import GridSearchCV,KFold,cross_val_score,train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from sklearn.metrics import r2_score
import joblib
import pickle

# Recursive feature elimination
from sklearn.feature_selection import RFE

from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://{}:{}@localhost/{}'.format('root','madhu123','startup_db'))
#startup = pd.read_csv(r"C:\Users\madhu\OneDrive\Desktop\360 DigiTMG\DataScience\DATA SETS\Datasets_MLR\50_Startups.csv")

#startup.to_sql('startup_tbl',con = engine,chunksize=1000,if_exists='replace',index = False)
############################## Data Import ###############################################################

startup = pd.read_csv(r"C:\Users\madhu\OneDrive\Desktop\360 DigiTMG\DataScience\DATA SETS\Datasets_MLR\50_Startups.csv")
startup.columns = 'RD','admin','marketing','state','profit'

#################################### Descriptive Statistics ##############################################

startup.head()

startup.describe()
startup['state'].value_counts()
startup.isna().any()
startup.corr()
sns.pairplot(startup)
sns.heatmap(startup.corr(),annot = True, cmap = "YlGnBu") 

report = sweetviz.analyze(startup)
report.show_html('startup.html')

startup.plot(kind = 'box',subplots = True, sharey = False,figsize = (15,6))
plt.subplots_adjust(wspace=0.75)
plt.show()
##########################################################################################################
X = startup.drop(['profit'],axis =1)
Y = startup.profit
########################################## Data Preprocessing ##############################################
num_cols = X.select_dtypes(include=['int64','float64']).columns
categ_cols = X.select_dtypes(include = ['object']).columns

num_pipeline = Pipeline([('winsor',Winsorizer(capping_method='iqr',fold = 1.5,tail = 'both')),
                         ('imputer',SimpleImputer(strategy='mean')),
                         ('scaling',MinMaxScaler())
                         ])

categ_pipeline = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                           ('encoding',OneHotEncoder(drop= 'first'))
                           ])

preprocess_pipeline = ColumnTransformer([('categ',categ_pipeline,categ_cols),
                                ('num',num_pipeline,num_cols)
                                ],remainder='passthrough')

preprocess = preprocess_pipeline.fit(X)
joblib.dump(preprocess,'preprocess.pkl')
preprocess = joblib.load('preprocess.pkl')

cleandata = pd.DataFrame(preprocess.transform(X),columns=preprocess.get_feature_names_out())
##########################################################################################################
heat = pd.concat([cleandata,Y],axis =1)
sns.heatmap(heat.corr(),annot = True, cmap = "YlGnBu") 
############################ multicolinearity ############################################################
# marketing & RD = 0.72
# state_florida & state_newyork = -0.49
########################### corelation with profit #######################################
# newyork = 0.031
# state_florid = 0.12
# RD = 0.97
# marketing = 0.75
########################################## Model Building with constant ##############################################
P = add_constant(cleandata)
basemodel = sm.OLS(Y, P).fit()
basemodel.summary()
predicted = basemodel.predict(P)

test_resid  = Y - predicted
# RMSE value for train data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


vif = pd.Series([variance_inflation_factor(P.values,i) for i in range(P.shape[1])], index = P.columns)
vif

# Based on the varience influence factor we have seen that the constant 
# has higher the multicolinearity so i am checking without constant.
# Tune the model by verifying for influential observations
sm.graphics.influence_plot(basemodel)
##########################################################################################################

########################################## Model Building without constant ##############################################
basemodel1 = sm.OLS(Y, cleandata).fit()
basemodel1.summary()
predicted1 = basemodel1.predict(cleandata)
'''
[1] R² is computed without centering (uncentered) since the model does not contain a constant.
[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
'''

test_resid1  = Y - predicted1
# RMSE value for train data 
test_rmse1 = np.sqrt(np.mean(test_resid1 * test_resid1))
test_rmse1

vif = pd.Series([variance_inflation_factor(cleandata.values,i) for i in range(cleandata.shape[1])], index = cleandata.columns)
vif

# Without constant the root mean square error is high so add constant and remove other variables.
##########################################################################################################

########################################## Model Building with constant remove variables based on corr coeficient ##############################################
P = add_constant(cleandata)
# droping 'categ__state_Florida'

P.drop('categ__state_Florida',axis =1,inplace = True)

basemodel2 = sm.OLS(Y, P).fit()
basemodel2.summary()
predicted2 = basemodel2.predict(P)


test_resid2  = Y - predicted2
# RMSE value for train data 
test_rmse2 = np.sqrt(np.mean(test_resid2 * test_resid2))
test_rmse2

vif = pd.Series([variance_inflation_factor(P.values,i) for i in range(P.shape[1])], index = P.columns)
vif

##########################################################################################################

########################################## Model Building with constant remove variables based on corr coeficient ##############################################
P = add_constant(cleandata)
# droping 'categ__state_Florida', num__admin

P.drop(['categ__state_Florida','num__admin'],axis =1,inplace = True)

basemodel3 = sm.OLS(Y, P).fit()
basemodel3.summary()
predicted3 = basemodel3.predict(P)

test_resid3  = Y - predicted3
# RMSE value for train data 
test_rmse3 = np.sqrt(np.mean(test_resid3 * test_resid3))
test_rmse3

vif = pd.Series([variance_inflation_factor(P.values,i) for i in range(P.shape[1])], index = P.columns)
vif
# variance_inflation_factor are with in the range but probability still showing multicolinearity.

##########################################################################################################

########################################## Model Building with constant remove variables based on corr coeficient ##############################################
P = add_constant(cleandata)
# droping 'categ__state_Florida', num__admin,categ__state_New York

P.drop(['categ__state_Florida','num__admin','categ__state_New York'],axis =1,inplace = True)

basemodel4 = sm.OLS(Y, P).fit()
basemodel4.summary()
predicted4 = basemodel4.predict(P)

test_resid4  = Y - predicted4
# RMSE value for train data 
test_rmse4 = np.sqrt(np.mean(test_resid4 * test_resid4))
test_rmse4

vif = pd.Series([variance_inflation_factor(P.values,i) for i in range(P.shape[1])], index = P.columns)
vif
# variance_inflation_factor are with in the range but probability still showing multicolinearity.

sm.graphics.influence_plot(basemodel4)
# obseravations 45 , 46, 49,19 are widely.
##########################################################################################################

########################################## Model Building with constant remove variables based on corr coeficient ##############################################
P = add_constant(cleandata)
# droping 'categ__state_Florida', num__admin,categ__state_New York

P.drop(['categ__state_Florida','num__admin','categ__state_New York'],axis =1,inplace = True)

P = P.drop(P.index[[45,48,19,46,49,14]])
Y = Y.drop(Y.index[[45,48,19,46,49,14]])


basemodel5 = sm.OLS(Y, P).fit()
basemodel5.summary()
predicted5 = basemodel5.predict(P)

test_resid5  = Y - predicted5
# RMSE value for train data 
test_rmse5 = np.sqrt(np.mean(test_resid5 * test_resid5))
test_rmse5

#Y = startup.profit
##########################################################################################################

########################################## Testing on test and train data ##############################################

X_train, X_test, Y_train, Y_test = train_test_split(P, Y, 
                                                    test_size = 0.2, random_state = 0) 

## Build the best model Model building with out cv
model = sm.OLS(Y_train, X_train).fit()
model.summary()

# Predicting upon X_train
ytrain_pred = model.predict(X_train)
r_squared_train = r2_score(Y_train, ytrain_pred)
r_squared_train

# Train residual values
train_resid  = Y_train - ytrain_pred
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse


# Predicting upon X_test
y_pred = model.predict(X_test)

# checking the Accurarcy by using r2_score
r_squared = r2_score(Y_test, y_pred)
r_squared

# Test residual values
test_resid  = Y_test - y_pred
# RMSE value for train data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse
##########################################################################################################

########################################## Hyper parameter tuning ##############################################

## Scores with Cross Validation (cv)
# k-fold CV (using all variables)
lm = LinearRegression()

## Scores with KFold
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

scores = cross_val_score(lm, X_train, Y_train, scoring = 'r2', cv = folds)
scores   


## Model building with CV and RFE

# step-1: create a cross-validation scheme
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 9))}]


# step-3: perform grid search
# 3.1 specify model
# lm = LinearRegression()
lm.fit(X_train, Y_train)

# Recursive feature elimination
rfe = RFE(lm)

# 3.2 call GridSearchCV()
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring = 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score = True)      

# fit the model
model_cv.fit(X_train, Y_train)     

cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results

# plotting cv results
plt.figure(figsize = (16, 6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc = 'upper left')

# train and test scores get stable after 3rd feature. 
# we can select number of optimal features more than 3

model_cv.best_params_

model_cv.best_params_

cv_lm_grid = model_cv.best_estimator_
cv_lm_grid

## Saving the model into pickle file
pickle.dump(cv_lm_grid, open('startup.pkl', 'wb'))

##########################################################################################################

############################### Deployment Part #########################################################################

startup = pd.read_csv(r"C:\Users\madhu\OneDrive\Desktop\360 DigiTMG\DataScience\DATA SETS\Datasets_MLR\50_Startups.csv")
startup.columns = 'RD','admin','marketing','state','profit' # for new data no need to mention about profit



model1 = pickle.load(open('startup.pkl','rb'))
preprocess = joblib.load('preprocess.pkl')

startup = startup.drop(['profit'],axis =1) # for new data this step is no need.

cleandata = pd.DataFrame(preprocess.transform(startup),columns=preprocess.get_feature_names_out())
cleandata = add_constant(cleandata)
cleandata.drop(['categ__state_Florida','num__admin','categ__state_New York'],axis =1,inplace = True)

prediction = pd.DataFrame(model1.predict(cleandata), columns = ['Profit'])

prediction
