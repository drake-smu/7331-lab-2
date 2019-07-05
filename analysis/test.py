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
import matplotlib.pyplot as plt

#%%
adults = pd.read_csv("data/adult-training.csv",names=['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label'],  
        skipinitialspace = True)
adults_test = pd.read_csv('data/adult-training.csv',
    names=['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label'], 
        skipinitialspace = True)


#%%
train_data = adults.drop('label',axis=1)

test_data = adults_test.drop('label',axis=1)

data = train_data.append(test_data)

label = adults['label'].append(adults_test['label'])

#%%
data.head()

#%%
full_dataset = adults.append(adults_test)

#%%
data_binary = pd.get_dummies(data)

data_binary.head()

#%%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_binary,label,)

#%%
performance = []
#%%
# LogisticRegression
from sklearn.linear_model import LogisticRegression


logClassifier = LogisticRegression()

#%%
logClassifier.fit(x_train,y_train)
train_score = logClassifier.score(x_train,y_train)
test_score = logClassifier.score(x_test,y_test)

print(f'LogisticRegression : Training score - {round(train_score,6)} - Test score - {round(test_score,6)}')

performance.append({'algorithm':'LogisticRegression', 'training_score':round(train_score,6), 'testing_score':round(test_score,6)})

#%%
from sklearn.neighbors import KNeighborsClassifier

knn_scores = []

#%%

train_scores = []
test_scores = []

for n in range(1,20,2):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train,y_train)
    train_score = knn.score(x_train,y_train)
    test_score = knn.score(x_test,y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f'KNN : Training score - {train_score} -- Test score - {test_score}')
    knn_scores.append({'algorithm':'KNN', 'training_score':train_score})
    
#%%    
plt.scatter(x=range(1, 20, 2),y=train_scores,c='b')
plt.scatter(x=range(1, 20, 2),y=test_scores,c='r')

plt.show()

#%%
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

knn.score(x_train,y_train)

train_score = knn.score(x_train,y_train)
test_score = knn.score(x_test,y_test)

print(f'K Neighbors : Training score - {train_score} - Test score - {test_score}')

performance.append({'algorithm':'K Neighbors', 'training_score':train_score, 'testing_score':test_score})

#%%
