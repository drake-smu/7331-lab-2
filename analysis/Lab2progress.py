#%%
# ms-python.python added
import os
try:
	os.chdir('~/7331-lab-2') 
	print(os.getcwd())
except:
	pass

#%% [markdown]
#  # Data Mining 7331 - Summer 2019
#  ## Lab 2 - Logistic Regression and Support Vector Machines
# 
#  ### Carson Drake, Che Cobb, David Josephs, Andy Heroy
# 
#
# <a id="top"></a>
# ## Table of Contents
# * <a href="#business">Section 1: Business Understanding</a>
#   * <a href="#business1">Section 1.1: Data Description</a>
#   * <a href="#business2">Section 1.2: Data Potential</a>
# * <a href="#understanding">Section 2: Data Understanding</a>
#   * <a href="#understanding1">Section 2.1: Variable Description</a>
#   * <a href="#understanding2">Section 2.2: Data Quality</a>
#   * <a href="#understanding3">Section 2.3: Simple Statistics</a>
#   * <a href="#understanding4">Section 2.4: Interesting Visualizations</a>
# * <a href="#preparation">Section 3: Data Preparation</a>
#   * <a href="#preparation1">Section 3.1:Part 1</a>
#   * <a href="#preparation2">Section 3.2:Part 2</a>
# * <a href="#modeling">Section 4: Modeling and Evaluation</a>
#   * <a href="#modeling1">Section 4.1:Part 1</a>
#   * <a href="#modeling2">Section 4.2:Part 2</a>
#   * <a href="#modeling3">Section 4.3:Part 3</a>
#       * <a href="#modeling3_1">Task 1:Classification 1</a>
#           * <a href="#modeling3_1_1">Logistic Regression</a>
#           * <a href="#modeling3_1_2">Random Forest</a>    
#           * <a href="#modeling3_1_3">KNN</a>
#       * <a href="#modeling3_2">Task 2:Classification 2</a>
#           * <a href="#modeling3_2_1">Logistic Regression</a>
#           * <a href="#modeling3_2_2">Random Forest</a>    
#           * <a href="#modeling3_2_3">Naive Bayes</a> 
#           * <a href="#modeling3_2_4">Stochastic Gradient Descent:</a> 
#   * <a href="#modeling4">Section 4.4:Part 4</a>
#   * <a href="#modeling5">Section 4.5:Part 5</a>
#   * <a href="#modeling6">Section 4.6:Part 6</a>
#   * <a href="#deployment">Section 4.7:Deployment</a>
#   * <a href="#exceptional">Section 4.8:Exceptional Work</a>
#

#%% [markdown]
# <a id="business"></a> <a href="#top">Back to Top</a>
#  ## Section 1: Business Understanding
# <a id="business1"></a> <a href="#top">Back to Top</a>
#  ### Section 1.1: Data Description
# 
#  Describe the purpose of the data set you selected.
#  We chose this dataset from the UCI's machine learning repository for its categorical
#  predictive attributes.  It contains 1994 Census data pulled from the US Census
#  database.  The prediction task we've set forth is to predict if a persons
#  salary range is >50k in a 1994, based on the various categorical/numerical
#  attributes in the census database. The link to the data source is below:
# 
#  https://archive.ics.uci.edu/ml/datasets/census+income
#
# <a id="business2"></a> <a href="#top">Back to Top</a>
#  ### Section 1.2: Data potential
#  
#  Describe how you would define and measure the outcomes from the dataset.
#  (That is, why is this data important and how do you know if you have mined
#  useful knowledge from the dataset? How would you measure the effectiveness of
#  a good prediction algorithm? Be specific.)
# 
#  The main benefit of this data is to be able to predict a persons salary range
#  based on factors collected around each worker in 1994.  With that insight, we
#  can look at a persons, age, education, marital status, occupation and begin to
#  explore the relationships that most influence income.  We'd like to find:
#    * What factors are the strongest influence of a how much many they will
#      make.
#    * What age groups show the largest amount of incomes over >50k?  aka, what
#      years of our life should we be working hardest in order to make the most
#      money.
#    * Does where you come from influence your income? (native country)
#%% [markdown]
# <a id="understanding"></a> <a href="#top">Back to Top</a>
#  ## Section 2: Data Understanding
# <a id="understanding1"></a> <a href="#top">Back to Top</a>
#  ### Section 2.1: Variable Description
#  
#  Describe the meaning and type of data for each attribute
#  Here we will discuss each attribute and give some description about its ranges.
# 
# 
#  Categorical - Description
#  #### Categorical Attributes
#  * workclass - Which business sector do they work in?
#  * education - What level of education received?
#  * marital_status - What is their marriage history
#  * occupation - What do they do for a living
#  * relationship - Family member relation
#  * race - What is the subjects race
#  * gender - What is the subjects gender
#  * native_country - Where is the subject originally from
#  * income_bracket - Do they make over or under 50k/year
# 
#  #### Continuous Attributes
#  * age - How old is the subject?
#  * fnlwgt - Sampling weight of observation
#  * education_num - numerical encoding of education variable
#  * capital_gain - income from investment sources, separate from wages/salary
#  * capital_loss - losses from investment sources, separate from wages/salary
#  * hours_per_week - How many hours a week did they work?
# 
#
# <a id="understanding2"></a> <a href="#top">Back to Top</a>
#  ### Section 2.2: Data Quality
#  Verify data quality: Explain any missing values, duplicate data, and outliers.
#  Are those mistakes? How do we deal with these problems?
# 
#  In the next code section we will import our libraries and data, then begin looking at
#  missing data, duplicate data, and outliers.
# %%
# Add library references
import pandas as pd
import numpy as np
import seaborn as sns
#import plotly.plotly as py
#import plotly.graph_objs as go
import matplotlib.pyplot as plt
import timeit
get_ipython().run_line_magic('matplotlib', 'inline')



# %%
df_headers = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income_bracket'
]
df_census = pd.read_csv("data/adult-training.csv",
    names=df_headers, 
    index_col=False)
# Input in case we want to combine the dataframes. 
# df_test = pd.read_csv("data/adult-test.csv",names = df_headers,skiprows=1)
# df_census = pd.concat([df_test, df_census], axis=0)

df_census.head(10)

# %% [markdown]
#  First, we'll start with looking at the head of the table to get a
#  feel for overall structure and the variables that we're working with. Followed
#  by a count of any missing values within the dataset.  We see that our data has
#  no missing values which is great under most circumstances, but we also found
#  that instead of marking the data with an NA, they did so with a "?.  Our first
#  order of business is to replace those values.  We found counts of ? values in
#  WorkClass, Occupation, and native country.  For now, we'll replace them with
#  "Other"
#
#

# %%
print("Structure of data:\n",df_census.shape,"\n")
print("Count of missing values:\n",df_census.isnull().sum().sort_values(ascending=False),"\n")
print("Count of ? values in workclass: " ,df_census.loc[df_census.workclass == ' ?', 'workclass'].count())
print("Count of ? values in occupation: ", df_census.loc[df_census.occupation == ' ?', 'occupation'].count())
print("Count of ? values in native_country: ", df_census.loc[df_census.native_country == ' ?', 'native_country'].count())

# %% [markdown]
#  While our missing values count is very low, we now must change
#  all the ? entries to other in order not cause further errors.  We'll also be
#  grouping each individual native country into their respective continent.  We
#  feel that grouping as such will give us more insight into how U.S. immigrants
#  fare in the job market.  We'll also introduce a pair plot to look in the
#  visualization section to look for any outliers.  Which spoiler alert, it
#  doesn't look like we have any that cause great concern.

# %%
# Change income bracket values that have a . at end and remove space 
df_census = df_census.replace(to_replace=(' >50K.', ' >50K'),value='>50K')
df_census = df_census.replace(to_replace=(' <=50K.', ' <=50K'),value='<=50K')    
df_census = df_census.replace(to_replace=(' United-States', ' Honduras', ' Mexico',' Puerto-Rico',' Canada', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua', ' Guatemala', ' El-Salvador' ),value='North America')
df_census = df_census.replace(to_replace=(' Cuba', ' Jamaica', ' Trinadad&Tobago', ' Haiti', ' Dominican-Republic' ),value='Caribbean')
df_census = df_census.replace(to_replace=(' South', ' Cambodia',' Thailand',' Laos', ' Taiwan', ' China', ' Japan', ' India', ' Iran', ' Philippines', ' Vietnam', ' Hong'),value='Asia')
df_census = df_census.replace(to_replace=(' England', ' Germany', ' Portugal', ' Italy', ' Poland', ' France', ' Yugoslavia',' Scotland', ' Greece', ' Ireland', ' Hungary', ' Holand-Netherlands'),value='Europe') 
df_census = df_census.replace(to_replace=(' Columbia', ' Ecuador', ' Peru'),value='South America')
df_census = df_census.replace(to_replace=(' ?'),value='Other') 

# encoding into 1 and zero variables for income_bracket. 
# df_census['income_bracket'] = df_census['income_bracket'].apply(lambda x: 1 if x=='>50K' else 0)

# %% [markdown]
# <a id="understanding3"></a> <a href="#top">Back to Top</a>
#  ### Section 2c: Simple Statistics
#
#  #### Visualize appropriate statistics (e.g., range, mode, mean, median, variance, counts) for a subset of attributes. Describe anything meaningful you found from this or if you found something potentially interesting.
#
#  Now that our data has been cleansed of any obvious errors, it's time to look at
#  the statistics behind our continuous data in order to look for any other
#  errors in the data we might have missed.  We also can get a look at how many
#  variables each of our categorical attributes carry with them.  This will be
#  useful down the line when we start grouping items for our basic EDA charts we
#  would like to produce.

# %%
for i in df_headers:
    
    print(i, 
    "type: {}".format(df_census[i].dtype),
    "# unique: {}".format(df_census[i].nunique()),
    sep="\n  ", end="\n\n")
    
print("Summary Statistic's:\n",round(df_census.describe().unstack(),2),"\n")



# %%
education_categories = list(df_census.education.unique())
print(df_census.groupby(['education','gender'])['gender'].count().unstack())


# %%
secondary = [
    'education',
    'gender',
    'race',
    'marital_status',
    'relationship',
    'native_country',
    'workclass'
    ]
for i in secondary:
    print(df_census.groupby([i,'income_bracket'])[i].count().unstack(), end="\n\n")


# %% [markdown]
# the categories that we've analyzed.  One category of capital_gain has some
# very large numbers, but we might attribute that to massive investments made by
# one individual.  After exploring further, alot of the values are 99,999. Which
# we assume to be a cap on what's reported for capital gains.  We did find that
# most of the occupations showing such capital growth was mostly executives.  So
# we're not surprised to see the higher numbers here and won't change the data.
#
# We also wanted to get a look at some of the educational categories by gender
# and income bracket to look for interesting statistics there.  We noticed that
# males tend to have more education across all education levels.  We also found that
# when looking at income bracket and education, a bachelors degree doesn't put
# you in a better place to make over 50k a year.  In fact, the only categories
# that did have a higher count in the >50k income bracket were Doctorate,
# Masters, or a professional school. 

# %% [markdown]
#
# <a id="understanding4"></a> <a href="#top">Back to Top</a>
#  ### Section 2d: Interesting Visualizations
#
#  #### Visualize the most interesting attributes (at least 5 attributes, your opinion on what is interesting). Important: Interpret the implications for each visualization. Explain for each attribute why the chosen visualization is appropriate.
#
#  Now we can start analyzing different attributes to see if anything stands out
#  to us.  To start we'll begin with some histograms of the numerical attributes
#  in order to look at the ranges again and check for skew.  We'll also look at
#  some box plots of gender and marital status to continue our exploration into
#  those categories.

# %%
#Histogram charts
sns.set_style('whitegrid')
df_num = df_census.select_dtypes(include=['float64'])
df_census.hist(figsize =(14,12))


# %% [markdown]
#  The histograms show us all things we expect to see from the
#  numerical categories.  Most of the workforce is from 20 to 50.  Educational
#  lidsmitations look to have the largest difference between 8th - 9th grade.
#  Implying that high school drop out rates are a factor in the dataset.   Hours
#  per week also exhibited a large distribution around 40 hours a week, which
#  fits common conception of American work hours.  fnl weight also showed some
#  strange values in the upper ranges of the dataset, but seeing as its not
#  going to be an area of focus for this analysis, we'll omit any changes here.

# %%
## boxplots of income by gender dist.
sns.set_style('whitegrid')
sns.countplot(x='income_bracket',
    hue='gender',
    data=df_census,
    palette='RdBu_r')

# %% [markdown]
#  This bar chart shows us the differences in male and female income based on
#  gender.  We see counts are much higher in both income brackets for males.
#  Suggesting that in 1994, the American workforce sampled had more men than
#  women in the workforce.  In the >50k income bracket, males showed an even
#  higher difference between their female counterparts, suggesting that males
#  dominate that income bracket more so than those in the <=50 income bracket.
#

# %%
## by marital status
sns.set_style('whitegrid')
sns.countplot(x='income_bracket',
    hue='marital_status',
    data=df_census,
    palette='RdBu_r')

# %% [markdown]
#
#  This bar chart represents income bracket by marital status. Interesting to
#  see a few things, first off the <=50k income bracket highest counts come from
#  the "Never-married" status.  This suggests that marriage does in fact come
#  with alot of financial benefit, as you can see is relevant on the other half
#  of the chart.  As married couples far outmatch any other category counts in
#  the >50k income bracket.  We can confirm this again as most of the divorced,
#  separated, or widowed people are located in the lower income bracket.
#  Suggesting that, if you want to make over 50k, you might want to get yourself
#  a partner and keep them! For our next chart, Lets split up the age groups in
#  bins of 10 years, and see what kind of income differences we see.
#
# %%
df_age = df_census.loc[:,['gender', 'age', 'income_bracket', 'hours_per_week']]
conditions = [
    (df_age['age'] < 20),
    (df_age['age'] < 30),
    (df_age['age'] < 40),
    (df_age['age'] < 50),
    (df_age['age'] < 60),
    (df_age['age'] < 70),
    (df_age['age'] < 110)]
choices = ['10-20', '20-30', '30-40','40-50','50-60','60-70','70-110']
df_age['age_group'] = np.select(conditions, choices, default='70-110')

sns.set_style('whitegrid')
sns.countplot(x='age_group',
    hue='income_bracket',
    data=df_age,
    palette='RdBu_r',
    order=choices)

# %% [markdown]
#
#  The first thing we're drawn too is that not many 10-20 year olds are making
#  over 50k!  What a surprise.  Its interesting how the two income groups tend
#  to converge once age groups get to the 40-50 range, but then both steadily
#  decline afterwards.  This follows suit with the average retirement age in
#  America of 62 years old.  But the largest jump in those in the >50k group
#  looks to happen around age 30 to 40.  Suggesting that if you're not clearing
#  that mark by 40, then chances are its going to get a harder to do so from
#  then on. Next we'll, analyze means of hours worked per the
#  education category.

#%%

df = df_census[['hours_per_week', 'education','education_num']].groupby('education').apply(lambda x: x.mean())
df.sort_values('education_num', inplace=True)
df.reset_index(inplace=True)

# Draw plot
fig, ax = plt.subplots(figsize=(10,10), dpi= 80)
ax.hlines(y=df.index, xmin=30, xmax=50, color='gray', alpha=0.7, linewidth=1, linestyles='dashdot')
ax.scatter(y=df.index, x=df.hours_per_week, s=75, color='firebrick', alpha=0.7)

# Title, Label, Ticks and Ylim
ax.set_title('Dot Plot for hours per week by education level', fontdict={'size':22})
ax.set_xlabel('hours per week')
ax.set_yticks(df.index)
ax.set_yticklabels(df.education.str.title(), fontdict={'horizontalalignment': 'right'})
ax.set_xlim(30, 50)
plt.show()

# %% [markdown]
#
# <a id="preparation"></a> <a href="#top">Back to Top</a>
# ## Section 3: Data Preparation:Part 1
#
# <a id="preparation1"></a> <a href="#top">Back to Top</a>
# ### Section 3.1 Part 1:
#
# Define and prepare your class variables. Use proper variable representations
# (int, float, one-hot, etc.). Use pre-processing methods (as needed) for
# dimensionality reduction, scaling, etc. Remove variables that are not
# needed/useful for the analysis.
#
# We've built a seperate py file that does all the pre-processing and will
# automatically clean and generate our dataframes for classification.  We will
# import it as lab_db from the dataBuilding py file.  It has all the necessary
# functions in order to build our data frames to be analyzed in the modeling
# section below.  Below is a list of the basic data cleaning and variable
# manipulation
#
# ### Data Specific cleaning
# 1.  Reduced education levels to 3 levels of No Diploma, Associates, and Diploma
# 2.  stripped any spaces off the leading or trailing characters
# 3.  Segmented country of origin to continent
# 4.  Encode the income_bracket target as binary
#
#
# ### Preprocessing
# For the continous variables, we will impute the median for any missing values
# and then use the StandardScaler to scale all the value's to a normalized
# range. Categorical attributes are transformed via sklearns "OneHotEncoder."
# This functions assigns a binary column to each category for every attribute.
# Currently we've set it to ignore any unknown variables ie - missing value's.  
# 
# Our data cleaning and encoding scripts are shown in the next cell: 
#

# %%
%load analysis/dataBuilding.py
# %%
# Data Import
#
from analysis import dataBuilding as lab_db

# Assign Default Vales for Columns
cat_cols,cont_cols,drop_cols = lab_db.cat_cols,lab_db.cont_cols,lab_db.drop_cols

# Drop Columns (if any)
X,y = lab_db.build_df(drop_cols)

# Transform continuous cols to scaled versions
# Transform categorical cols to Encoded Cols
trans = lab_db.build_transform(cont_cols,cat_cols)

#%%
# Execute Transforms specified above on all X data
X_processed = trans[1].fit_transform(X)
enc_headers = trans[1].named_transformers_['cat'].named_steps['onehot'].get_feature_names()
new_headers = np.concatenate((cont_cols,enc_headers))
#%%
# Split processed X data into training and test sets.
# Also separate y (label) data into training and test sets.
X_train, X_test, y_train, y_test = lab_db.split_df(X_processed,y,0.2)



#%% [markdown]
#
# <a id="preparation2"></a> <a href="#top">Back to Top</a>
# ### Section 3.2 Part 2:
#
# Describe the final dataset that is used for classification/regression (include
# a description of any newly formed variables you created).
#
# 
# Our final dataset shape

print('Number of observations in the training data:', len(X_train))
print('Number of observations in the test data:',len(X_test))
#
# For the final shape of our dataset with the split, we have 39073 instances and
# 62 attributes.  The added attributes come from the one hot encoding that
# splits the categories in to their own binary response for faster code
# excecution.  
#

# %% [markdown]
#
# <a id="modeling"></a> <a href="#top">Back to Top</a>
# ## Section 4: Modeling and Evaluation:
#
#
# <a id="modeling1"></a> <a href="#top">Back to Top</a>
# ### Section 4.1 Part 1:
#
# Choose and explain your evaluation metrics that you will use (i.e., accuracy,
# precision, recall, F-measure, or any metric we have discussed). Why are the
# measure(s) appropriate for analyzing the results of your modeling? Give a
# detailed explanation backing up any assertions.
#
#
# Lets take the time to explain some of the precision outputs.
# * precision - this is the ratio of the number of true positives and false
#   positives.
# * recall - this is the ratio of the number of true positives and false
#   negatives
# * f1-score - the harmonic mean of the precision and recall.
# * support - occurances in each class
# * accuracy - count of predictions where the predicted value equals the actual
#   value
# * Log Loss - the negative log-likelihood of correct classification given the
#   classifier prediction.
#
#
# We settled on accuracy and F-1 score as our two metrics for tracking
# performance. Accuracy is a desired metric for classification as you're always
# trying to improve your models ability to predict. F1-Score a balanced
# mean between precision and recall. Our initial analysis shows we still have a
# fair amount of false positives showing an uneven balanced class, which is why
# we will focus on F1 as well for this analysis. We also will use a normalized
# confusion matrix in order to gain insight into how each of our models are
# performing. Lastly, We will do a statistical analysis comparing these metrics
# later in the report.
#
#
#
# <a id="modeling2"></a> <a href="#top">Back to Top</a>
# ### Section 4.2 Part 2:
#
# Choose the method you will use for dividing your data into training and
# testing splits (i.e., are you using Stratified 10-fold cross validation?
# Why?). Explain why your chosen method is appropriate or use more than one
# method as appropriate. For example, if you are using time series data then you
# should be using continuous training and testing sets across time.
#
# We chose in this analysis to use the train/test split method of cross validation. Due to 
# our sufficiently large data set, we only need to do one fold. Because we are predicting on categorical data, a
# and every row is unique, we just used simple sklearn sampling of our dataset to generate the two splits.
# We can see the size of the splits in the cell above


# %% [markdown] 
#
# <a id="modeling3"></a> <a href="#top">Back to Top</a>
# ### Section 4.3 Part 3:
#
# Create three different classification/regression models for each task (e.g.,
# random forest, KNN, and SVM for task one and the same or different algorithms
# for task two). Two modeling techniques must be new (but the third could be SVM
# or logistic regression). Adjust parameters as appropriate to increase
# generalization performance using your chosen metric. You must investigate
# different parameters of the algorithms!
#
#
# <a id="modeling3_1"></a> <a href="#top">Back to Top</a>
# ### Task 1:  Classification of making > or <= 50k
#
# Our first task is to determine a persons income bracket
# by way of 3 different classification models. Our first attempt will be to
# create a logistic regression model. We will follow that with other
# classification methods such as Random Forest, and K-nearest neighbor.
#

#%%
def plot_confusion_matrix(y_true, y_pred, 
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# %% [markdown]
#
# <a id="modeling3_1_1"></a> <a href="#top">Back to Top</a>
#
# ### Logistic Regression
#
#
# Logistic regression (LR) is a classification algorithm thats used to predict
# the probability of our categorical dependent variable.  The basics behind LR
# is that it takes the output of a linear model and crams it into a logistic
# function to give it a probablity of 0 to 1 (but never equaling 0 or 1). We
# will implement this model as our first for Task 1.
#
#

#%%
# # Initialize performance array to store model performances for various 
# # models used in Lab 02.

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

start = timeit.default_timer()
performance = []
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


#%%
y_pred = logClassifier.predict(X_test)
plot_confusion_matrix(y_test,y_pred)
print(f'Logistic regression : accuracy score - {metrics.accuracy_score(y_test,y_pred)}')

print(f'Logistic regression : f1 score - {metrics.f1_score(y_test,y_pred)}')

stop = timeit.default_timer()
t = stop - start
performance.append({'algorithm':'LogisticRegressionT1',
    'accuracy':metrics.accuracy_score(y_test,y_pred),
    'f1 score':metrics.f1_score(y_test,y_pred),
    'observations' : len(y_test),
    'run time' : t})
#%% [markdown]
#
# Our Logistic regression gives us an accuracy of 84.8%, however suffered from a lower F1 score of 0.639. 
# This is likely due to the models high amount of false negatives, as we can see from the confusion matrix, the model gave us nearly
# as many false negatives as true positives. It was however very good at detecting true negatives.

# %% [markdown]
#
# <a id="modeling3_1_2"></a> <a href="#top">Back to Top</a>
# ### Random Forest
#
#
# One of the most commonly used classifier techniques is random forest, due to
# its very low bias and general stability when it comes to classification.  One
# method of optimizing a random forest model is to try different parameters to
# increase performance. Another method of doing so is by utilizing grid search
# to let random forrest decide which combination of hyperparameters would be best
# implemented in your model.  We chose this route as it saves both time and
# sanity when comparing so many different parameters.   
#
# We'll start with a baseline random forest for our starting position.  
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


clf =RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(f'Random Forest : Accuracy score - {metrics.accuracy_score(y_test, y_pred)}')

print(f'Random Forest : F1 score - {metrics.f1_score(y_test, y_pred)}')
#%%
plot_confusion_matrix(y_test, y_pred)

# %% [markdown]
#
# Our initial run shows a decent accuracy with a basis of n_estimators at 100.
# With an Accuracy of 84.7%. The F1 score offered a slight improvement over that
# of the simple logistic regression, which is visible in the slightly lowered
# amount of false negatives in the confusion matrix. One clear issue with this
# model howeverm which causes it to be poorly performing, is the relatively high
# rate of false positives. Thats a pretty good initial run, but improvements can
# be made.  Now we will implement a gridsearch over this random forrest to
# extract optimal hyperparameters for tuning our random forest.  Our chosen
# hyperparameter tuning features will be
#
# ** max_features **
#   * Max Features looks to optimaze the number of features to consider when
#     looking for a split
#
#
# ** n_estimators ** 
#   * n_estimators sets the number of tree's in a forest.  Adding more tree's
#     will increase your accuracy, but also make the training process very time
#     costly. 
#
#
# ** min_samples_leaf **
#   * This is the minimum number of samples required for a leaf node to be
#     created.  
#



# %%

# commented out the grid search so it would run faster.  We can reinput at final run time. 

#clf=RandomForestClassifier()
#kf=KFold(n_splits=3)
#max_features=np.array([1,2,3,4,5])
#n_estimators=np.array([25,50,100,150,200])
#min_samples_leaf=np.array([25,50,75,100])
#
#max_features=np.array([1,2,3,4,5])
#n_estimators=np.array([25,50,100,150,200])
#min_samples_leaf=np.array([25,50,75,100])
#
# param_grid=dict(n_estimators=n_estimators,max_features=max_features,min_samples_leaf=min_samples_leaf)
# grid=GridSearchCV(estimator=clf,param_grid=param_grid,cv=kf)
# gris=grid.fit(X_train,y_train)

# print("Best",gris.best_score_)
# print("params",gris.best_params_)

# %% 
clf =RandomForestClassifier(n_estimators=50,max_features=5,min_samples_leaf=50)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(f'Random Forest : Accuracy score - {metrics.accuracy_score(y_test, y_pred)}')
print(f'Random Forest : f1 score - {metrics.f1_score(y_test, y_pred)}')
# performance.append({'algorithm':'Random Forrest', 'testing_score':metrics.accuracy_score(y_test, y_pred)})
#  Commented out performance append because we just want to capture the last run.

## This gives you the name of the features that are important according to the RFC
feature_imp = pd.Series(clf.feature_importances_,index=new_headers).sort_values(ascending=False)
top_feat = feature_imp.nlargest(n=8)
feature_imp

# %%
# Creating a bar plot
sns.barplot(x=top_feat, y=top_feat.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

#%% [markdown] 
# 
# With these parameters, we have built improved our rate of false
# positives slightly, but drastically worsened our rate of false negatives:
#
#%%

plot_confusion_matrix(y_test, y_pred)

#%% [markdown]
#
# Let us continue adjusting the model.
# First, let us adjust the number of features, and allow for basically any number of features to be included in the final model. 
# This should help with the false negative problem we have been having.
#
#%%


clf =RandomForestClassifier(n_estimators=50,max_features=50,min_samples_leaf=50, n_jobs = -1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
plot_confusion_matrix(y_test, y_pred)
print(f'Random Forest : Accuracy score - {metrics.accuracy_score(y_test, y_pred)}')
print(f'Random Forest : f1 score - {metrics.f1_score(y_test, y_pred)}')

## This gives you the name of the features that are important according to the RFC
feature_imp = pd.Series(clf.feature_importances_,index=new_headers).sort_values(ascending=False)
top_feat = feature_imp.nlargest(n=8)
feature_imp
#%% [markdown]
#
# This is our best model so far. We have maintained the low rate of false
# positives, while also lowering the rate of false negatives to near its default
# level. Let us check out this models most important features:
#
# %%
# Creating a bar plot
sns.barplot(x=top_feat, y=top_feat.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

#%% [markdown] 
# 
# Let us now tune the model even further, raising the number of
# estimators, as well as lowering the the minumum samples per leaf:
#

start = timeit.default_timer()

clf =RandomForestClassifier(n_estimators=500,max_features=50,min_samples_leaf=10, n_jobs = -1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

stop = timeit.default_timer()
t = stop - start

plot_confusion_matrix(y_test, y_pred)
print(f'Random Forest : Accuracy score - {metrics.accuracy_score(y_test, y_pred)}')
print(f'Random Forest : F1 score - {metrics.f1_score(y_test, y_pred)}')
performance.append({'algorithm':'Random ForestT1', 
    'accuracy':metrics.accuracy_score(y_test, y_pred),
    'f1 score':metrics.f1_score(y_test,y_pred),
    'observations' : len(y_test),
    'run time' : t})



## This gives you the name of the features that are important according to the RFC
feature_imp = pd.Series(clf.feature_importances_,index=new_headers).sort_values(ascending=False)
top_feat = feature_imp.nlargest(n=8)
feature_imp

#%%
# A significant improvement! We have managed to slightly improve the false negative rate of the original random forest,
# while keepig our noticibeale false positive rate improvements. And look at that high F1 score! Lets check out what features this new
# super-accurate model suggests:
#%%

sns.barplot(x=top_feat, y=top_feat.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
#%% [markdown]
#
# Interestingly, this and a lot of the much cheaper models have very similar top
# features. Therefore, if the goal of our analysis was to simply determine the
# important features to make a simple model, we would want to choose one of the
# cheaper models with similar results. However, if we are going for accuracy,
# and low false positive and false negtive rates, this more expensive model is
# demonstrably superior 
# 
#
# %% [markdown]
#
# <a id="modeling3_1_3"></a> <a href="#top">Back to Top</a>
# ### KNN: K-nearest neighbors
#
# Next we have our KNN model using varying n_neighbor values. We will attempt to
# identify the optimal number of neighbors that allows for highest degree of
# accurancy while also being useful when tested on both the training and test
# set of data.
#
# %% 
start = timeit.default_timer()

knn_scores = []
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
    knn_scores.append({'algorithm':'KNN', 'training_score':train_score,'testing_score':test_score})



# %% [markdown]
#
# Visualizing the test vs training model scores helps us identify the the
# optimal n_neighbors level to train the model. We want to minimize the
# difference in performance between two sets while as well as optimize number of
# neighbors needed.  From the look of the scatterplot below, 5 nearest neighbors
# appears to be our best value for an even bias/variance tradeoff.  
# 
#

# %%
fig, ax = plt.subplots()
colors = ['tab:blue','tab:red']
for i,data in enumerate([train_scores,test_scores]):

    ax.scatter(x=range(1, 20, 2),y=data, c=colors[i])
# plt.scatter(x=range(1, 20, 2),y=train_scores,c='b',)
# plt.scatter(x=range(1, 20, 2),y=test_scores,c='r')
plt.style.use('seaborn-pastel')
plt.show()

#%% [markdown]
#
# Let us take a look at the accuracy and F1 scores for our best KNN model, as
# well as a confusion matrix:
#%%

knn = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

stop = timeit.default_timer()
t = stop - start

plot_confusion_matrix(y_test, y_pred)
print(f'KNN : Accuracy score - {metrics.accuracy_score(y_test, y_pred)}')
print(f'KNN : f1 score - {metrics.f1_score(y_test, y_pred)}')
performance.append({'algorithm':'KNNT1', 
    'accuracy':metrics.accuracy_score(y_test, y_pred),
    'f1 score':metrics.f1_score(y_test,y_pred),
    'observations' : len(y_test),
    'run time' : t})


#%% [markdown]
#
# Looking at the result of our knn test, it is not as well performing as the
# random forest. It is slower, has lower accuracy, and has an overall low  f1 score. This is likely due to the relatively high rate of false positives.
# 
#
# 

# %% [markdown]
#
# <a id="modeling3_2"></a> <a href="#top">Back to Top</a>
# ### Task 2:  Classification of male or female
#
#
# For Task 2, we decided to predict the gender of a person in the Census data.
# This was chosen due to the binary outcome of that column and it seemed like
# fun to predict.  On this task, we will classify the gender of the person using
# Logistic Regression, Random Forest, Naive Bayes, and Stochastic Gradient
# Descent
#
#

# %%
#
# Data Import for task 2
#
from analysis import dataBuilding as lab_db

# Assign Default Values for Columns
cat_cols2,cont_cols2,drop_cols = lab_db.cat_cols2,lab_db.cont_cols2,lab_db.drop_cols

# Drop Columns (if any)
X,y = lab_db.build_df2(drop_cols)

# Transform continuous cols to scaled versions
# Transform categorical cols to Encoded Cols
trans = lab_db.build_transform(cont_cols2,cat_cols2)

#%%
# Execute Transforms specified above on all X data
X_processed = trans[1].fit_transform(X)
enc_headers = trans[1].named_transformers_['cat'].named_steps['onehot'].get_feature_names()
new_headers = np.concatenate((cont_cols,enc_headers))
#%%
# Split processed X data into training and test sets.
# Also separate y (label) data into training and test sets.
X_train, X_test, y_train, y_test = lab_db.split_df(X_processed,y,0.2)


#%%
# # Initialize performance array to store model performances for various 
# # 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

start = timeit.default_timer()

logClassifier = LogisticRegression()

#%% [markdown]
# 
# <a id="modeling3_2_1"></a> <a href="#top">Back to Top</a>
# ### Logistic Regression
#
# First, we will be looking at a simple logistic regression model in order to predict whether a person, given demographic circumstances, is a 
# man or a woman. We are starting with logistic regression because it is fast, simple, and if it performs equally well as other models it is likely 
# the best model to choose, due to its cheapness and simplicity. Therefore it will act as a standard to compare to the other models

# %%
## Fit Logistic Classifier on training data
logClassifier.fit(X_train,y_train)
train_score = logClassifier.score(X_train,y_train)
test_score = logClassifier.score(X_test,y_test)

## Analyze how the model performed when tested against both 
## the data the model used for fit and test data. This helps 
## us identify overfitting.
print(f'LogisticRegression : Training score - {round(train_score,6)} - Test score - {round(test_score,6)}')

#%% [markdown]
# let us now look at a confusion matrix, accuracy, and F1 score fore this model:
#%%
y_pred = logClassifier.predict(X_test)

stop = timeit.default_timer()
t = stop - start

plot_confusion_matrix(y_test, y_pred)
print(f'Logistic Regression : Accuracy score - {metrics.accuracy_score(y_test, y_pred)}')
print(f'Logistic Regression : f1 score - {metrics.f1_score(y_test, y_pred)}')

performance.append({'algorithm':'LogisticRegressionT2',
    'accuracy':metrics.accuracy_score(y_test,y_pred),
    'f1 score':metrics.f1_score(y_test,y_pred),
    'observations' : len(y_test),
    'run time' : t})

#%% [markdown] 
# 
# This model is very interesting. Although it has a high rate of
# false positives, it has an exceedingly low rate of false negatives. Thus, it
# has a higher F1 score than it does accuracy. This model would be really useful
# in situations where where a false negative would be really bad but a false
# positive is fine
#
#
#  Let us now move on to the random forest. We will first run a grid search in
# parallel in order to find the proper parameters for this one. The random forest, while 
# a more expensive model than logistic regression, has a chance to not only give us a better preditcion/classification than 
# logisrtic regression, but also has a wonderful built in tool for viewing which estimators are the most important.
# This tool can be useful in deciding on further models, as well as just looking for which variables are most important and deserve a closer
# look or more data collection for future use.
# 
#
# %% [markdown] <a id="modeling3_2_2"></a> <a href="#top">Back to Top</a>
#
# ### Random Forest
#
# %%
#
#
#Che Forrest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

start = timeit.default_timer()

svcEstimator = RandomForestClassifier()
cv = 3

#compare various values of C, kernels (rbf vs linear vs poly),decision_function_shape (ovo vs ovr)
parameters = {'n_estimators': [100,250,500]
            , 'max_depth': [5,25,50,100]}

#Create a grid search object using the
from sklearn.model_selection import GridSearchCV
svcGridSearch = GridSearchCV(estimator=svcEstimator
                   , n_jobs=-1 # jobs to run in parallel
                   , verbose=1 # low verbosity
                   , param_grid=parameters
                   , cv=cv # KFolds = 3
                   , scoring='accuracy')

svcGridSearch.fit(X_train, y_train)
print("The best estimator based on F1 is ", svcGridSearch.best_estimator_)
best_rf = svcGridSearch.best_estimator_

# %%
#Create from best estimator search of random forest
clf=best_rf

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

stop = timeit.default_timer()
t = stop - start

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

## This gives you the name of the features that are important according to the RFC
feature_imp = pd.Series(clf.feature_importances_,index=new_headers).sort_values(ascending=False)
top_feat = feature_imp.nlargest(n=8)

# Creating a bar plot
sns.barplot(x=top_feat, y=top_feat.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

#%%

plot_confusion_matrix(y_test, y_pred)
print(f'Random Forest : Accuracy score - {metrics.accuracy_score(y_test, y_pred)}')
print(f'Random Forest : F1 score - {metrics.f1_score(y_test, y_pred)}')

performance.append({'algorithm':'Random ForestT2', 
    'accuracy':metrics.accuracy_score(y_test, y_pred),
    'f1 score':metrics.f1_score(y_test,y_pred),
    'observations' : len(y_test),
    'run time' : t})

#%% [markdown]
# 
# To create this model, we first ran a grid search in parallel to tune the hyperparameters to optimal values. We then took the best model and 
# delved deeper into it. This model boasts a very high accuracy, high F1 score, and a very low false negative (as well as very low false positive) rate.
# While improving accuracy, this model is highly computationally expensive. Therefore, this model would be used only in situations when accuracy is crucial.
# In situations in which we just want to find the best predictors, or just quickly get a decent model, any of the above models would suffice, however this model
# is extremely accurate, at the cost of relatively extreme computation time. The grid search is a powerful tool which can be used to tune models to perfection
# at a high computational cost
#
# TODO - Summarize Che forest run.  As its our best model
# <a id="modeling3_2_3"></a> <a href="#top">Back to Top</a>
# ### Naive Bayes
#
#%%
from sklearn.naive_bayes import GaussianNB

start = timeit.default_timer()

model = GaussianNB()
model.fit(X_train, y_train)
pred_y = model.predict(X_test)

stop = timeit.default_timer()
t = stop - start

plot_confusion_matrix(y_test,pred_y,normalize=True)
print("Naive Bayes : Accuracy:",metrics.accuracy_score(y_test, pred_y))

print("Naive Bayes : F1 score",metrics.f1_score(y_test, pred_y))

performance.append({'algorithm':'Naive BayesT2', 
    'accuracy':metrics.accuracy_score(y_test, pred_y),
    'f1 score':metrics.f1_score(y_test,pred_y),
    'observations' : len(y_test),
    'run time' : t})

#%% [markdown]
#
# TODO tune something here, i think we can get this F1 score to 90%
#
# Next we will try out a stochastic gradient descent model:
#
# <a id="modeling3_2_4"></a> <a href="#top">Back to Top</a>
#
# ### Stochastic Gradient Descent
#
# Stochastic Gradient Desent is a relative of gradient desent algorithm.  Where
# GD follows a step by step process through each observation, SGD shuffles
# its observations randomly.  Introducing less bias into the model.  
#
#%%
from sklearn.linear_model import SGDClassifier

start = timeit.default_timer()

clf = SGDClassifier(loss = "hinge", penalty="elasticnet", max_iter=5000, n_jobs = -1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

stop = timeit.default_timer()
t = stop - start

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred))

plot_confusion_matrix(y_test,pred_y,normalize=True)

performance.append({'algorithm':'SGD T2', 
    'accuracy':metrics.accuracy_score(y_test, y_pred),
    'f1 score':metrics.f1_score(y_test,y_pred),
    'observations' : len(y_test),
    'run time' : t})


#%% [markdown]
#
# TODO Maybe discuss this its basically just a SVM lol

# %% [markdown]
#
# <a id="modeling4"></a> <a href="#top">Back to Top</a>
# ### Section 4.4 Part 4:
#
# Analyze the results using your chosen method of evaluation. Use visualizations
# of the results to bolster the analysis. Explain any visuals and analyze why
# they are interesting to someone that might use this model.
#
# Below are the results from our various model runs over both tasks.  Judging
# from the intial results, it seems our Random forrests were the best performers
# on the accuracy side.  Yet the F1-score for Random ForestT1 was a bit lower.
# Because we widened our parameter search on the second random forest to a greater
# numer of estimators and a smaller minimum leaf node, it was able to yield a
# better accuracy due to increased tree creation. 
#
# TODO - finish describing why best model
#
# %% 
# Confidence interval
tperf = pd.DataFrame(performance)
tperf.round({'accuracy':5, 'f1 score':5})
tperf



# %% [markdown]
#
# <a id="modeling5"></a> <a href="#top">Back to Top</a>
# ### Section 4.5 Part 5:
#
# Discuss the advantages of each model for each classification task, if any. If
# there are not advantages, explain why. Is any model better than another? Is
# the difference significant with 95% confidence? Use proper statistical
# comparison methods. You must use statistical comparison techniques—be sure
# they are appropriate for your chosen method of validation as discussed in unit
# 7 of the course.
#
# To compare our models, we'll first analyze the variance of each binomial
# proportion confidence interval.  Then, assuming a Gaussian Distribution, we
# can apply confidence intervals to the classification accuracy.  This will give
# us our confidence intervals for accuracy.
#


# %% 
from math import sqrt
tnew = tperf
z = 1.96

for index_label, row_series in tnew.iterrows():

   tnew.at[index_label , 'confint'] = z * sqrt((row_series['accuracy'] * (1-row_series['accuracy']))/ row_series['observations'])
   tnew.at[index_label , 'upperinterval'] = row_series['accuracy'] + row_series['confint']
   tnew.at[index_label , 'lowerinterval'] = row_series['accuracy'] - row_series['confint']

tnew
# %% [markdown]
#
# All of our models came out with an accuracy of 82-87%.  The highest accuracy
# for Task 1 goes to random forestT1 with an accuracy of 87.2% [86.6%, 87.9%]
# Due to our extra hyperparameter tuning and grid search, this allowed for the
# greatest accuracy. Due to the fact none of its competing models lie with in
# the Random ForestT1's interval, we can reject the null hypothesis that they
# are the same and have 95% confidence that they are statistically different
# from the other models.  
#
# For Task 2, our best model was Logistic RegressionT2 with a classification
# accuracy of 84.6% [83.9%, 85.3%].  Because the confidence intervals for our
# log regT2 model do overlap with the random forestT2 results, we can't say
# there is a difference between the two.  Which gives us the discretion to
# choose whichever model we feel is best.  As logistic regression took about 1
# second to run and the random forest was nearly 200x that, we would select the
# Log RegT2 function due to time and cost savings for anyone who wanted to
# deploy that model.  

#
# <a id="modeling6"></a> <a href="#top">Back to Top</a>
# ### Section 4.6 Part 6:
#
# Which attributes from your analysis are most important? Use proper methods
# discussed in class to evaluate the importance of different attributes. Discuss
# the results and hypothesize about why certain attributes are more important
# than others for a given classification task.
#
#
#
# TODO - Bullshit on which attributes matter most for each task.  


# %% [markdown]
#
# <a id="deployment"></a> <a href="#top">Back to Top</a>
# ### Section 4.7 Deployment:
#
# How useful is your model for interested parties (i.e., the companies or
# organizations that might want to use it for prediction)? How would you measure
# the model's value if it was used by these parties? How would your deploy your
# model for interested parties? What other data should be collected? How often
# would the model need to be updated, etc.? 
#
#
# For our first task, targeting income, we forsee a multitude of possibilities
# for application. Whether it be for banks / credit card companies to improve
# their own models for how safe they feel lending a certain group of people
# money based on their financial circumstances.  It could also be applied for
# national survey's that people want to conduct.  Our model could help whittle
# down what factors they want to include in their updated model.  ie - which are
# the most significant variables and target those for analysis.  Another
# application could be for local governments, to help plan new schools, roads,
# child care and other facilities that benefit the population they reside in.  
#
#
# For our second task, targeting gender, One area where this model may prove
# useful is in collecting user information from websites.  Given many websites
# have categorical descriptions of their user data, it someone was say, in the
# healthcare sector and was attempting to survey the general public on
# cardiovascular disease.   Some people may not feel inclined to report their
# gender, but that information could prove useful to the researcher in order to
# classify the person properly.  Using our classification technique's, we could
# retrofit the model over their survey data and begin to do prediction on that.
#
#
# Measuring the models value would be done through a consistent monitoring of
# the algorithms Accuracy and F1-score as we've done in this project.  Giving
# the end user the ability to keep tabs on how well the model is predicting, as
# we add more data to it.  Deployment of the model, could be through a rest API
# service, an internal function built in the backend of whatever database system
# they have in place.  In terms of other data that could be useful to Census
# Data, we would suggest finding a resource that can provide, housing ownership,
# local economic key performance factors, zip codes to determine talent pools
# for who lives in what area.  The possibilities are large depending on what
# factors influence the customers business most.  As each business you would
# sell this model too would likely have their own idea's behind what factors
# make them sucssesful.  
#
#
#
# TODO - Madness inserted.  Feel free to add more. 
#

# %% [markdown]
#
# <a id="excpetional"></a> <a href="#top">Back to Top</a>
# ### Section 4.7 Exceptional Work:
#
# You have free reign to provide additional analyses. One idea: grid search
# parameters in a parallelized fashion and visualize the performances across
# attributes. Which parameters are most significant for making a good model for
# each classification algorithm?
#
# 
# TODO - Talk about Che's parallelized search
#
