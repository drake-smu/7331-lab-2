# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'labs/lab_02'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import timeit
#import plotly.plotly as py
#import plotly.graph_objs as go
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#%%
from analysis import dataBuilding as lab_db

## Assign Default Vales for Columns
cat_cols,cont_cols,drop_cols = lab_db.cat_cols,lab_db.cont_cols,lab_db.drop_cols




#%%

## Drop Columns (if any)
X,y = lab_db.build_df(drop_cols)

## Transform continuous cols to scaled versions
## Transform categorical cols to Encoded Cols
trans = lab_db.build_transform(cont_cols,cat_cols)

#%%
## Execute Transforms specified above on all X data
X_processed = trans[1].fit_transform(X)
enc_headers = trans[1].named_transformers_['cat'].named_steps['onehot'].get_feature_names()
new_headers = np.concatenate((cont_cols,enc_headers))
#%%
## Split processed X data into training and test sets.
## Also separate y (label) data into training and test sets.
X_train, X_test, y_train, y_test = lab_db.split_df(X_processed,y,0.2)

#%% 
## Initialize performance array to store model performances for various 
## models used in Lab 02.
performance = []
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
logClassifier = LogisticRegression()

#%%
## Fit Logistic Classifier on training data
logClassifier.fit(X_train,y_train)
train_score = logClassifier.score(X_train,y_train)
test_score = logClassifier.score(X_test,y_test)

## Analyze how the model performed when tested against both 
## the data the model used for fit and test data. This helps 
## us identify overfitting.
print(f'LogisticRegression : Training score - {round(train_score,6)} - Test score - {round(test_score,6)}')
performance.append({'algorithm':'LogisticRegression', 'training_score':round(train_score,6), 'testing_score':round(test_score,6)})

#%%
knn_scores = []
##
## Analyze KNN model using varying n_neighbor values. Identify 
## the optimal number of neighbors that allows for high degree 
## of accurancy while also being useful when tested on both 
## the training and test set of data.
train_scores = []
test_scores = []
for n in range(1,20,2):
    knn = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
    knn.fit(X_train,y_train)
    train_score = knn.score(X_train,y_train)
    test_score = knn.score(X_test,y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f'KNN : Training score - {train_score} -- Test score - {test_score}')
    knn_scores.append({'algorithm':'KNN', 'training_score':train_score})

#%%

## Visualizing the test vs training model scores helps us identify the 
## the optimal n_neighbors level to train the model. We want to minimize 
## the difference in permance between two sets while as well as optimize 
## number of neighbors needed.
fig, ax = plt.subplots()
colors = ['tab:blue','tab:red']
for i,data in enumerate([train_scores,test_scores]):

    ax.scatter(x=range(1, 20, 2),y=data, c=colors[i])
# plt.scatter(x=range(1, 20, 2),y=train_scores,c='b',)
# plt.scatter(x=range(1, 20, 2),y=test_scores,c='r')
plt.style.use('seaborn-pastel')
plt.show()

#%%
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#%%
## This gives you the name of the features that are important according to the RFC
feature_imp = pd.Series(clf.feature_importances_,index=new_headers).sort_values(ascending=False)
top_feat = feature_imp.nlargest(n=8)




#%%
# Creating a bar plot
sns.barplot(x=top_feat, y=top_feat.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

#%%
t2 = pd.Series(clf.feature_importances_).sort_values(ascending=False)
rf_scores = []
for i in range(1,60,5):
    _idx = t2.nlargest(n=i).index
    X_thin = X_train[:,_idx]
    X_thin_test = X_test[:,_idx]

    clf.fit(X_thin,y_train)

    y_pred=clf.predict(X_thin_test)
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    rf_scores.append({'num_features':i,'score':metrics.accuracy_score(y_test, y_pred)})

#%%
