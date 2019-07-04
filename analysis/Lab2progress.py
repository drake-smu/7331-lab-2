#%%
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'labs'))
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
# * <a href="#Business">Section 1: Business Understanding</a>
#   *<a href="#Business1">Section 1.1: Data Description</a>
#   *<a href="#Business2">Section 1.2: Data Potential</a>
# * <a href="#Understanding">Section 2: Data Understanding</a>
#   * <a href="#Understanding1">Section 2.1: Variable Description</a>
#   * <a href="#Understanding2">Section 2.2: Data Quality</a>
#   * <a href="#Understanding3">Section 2.3: Simple Statistics</a>
#   * <a href="#Understanding4">Section 2.4: Interesting Visualizations</a>
# * <a href="#distance">Measuring Distances</a>
# * <a href="#KNN">K-Nearest Neighbors</a>
# * <a href="#naive">Naive Bayes</a>

#%% [markdown]
# <a id="Business"></a> <a href="#top">Back to Top</a>
#  ## Section 1: Business Understanding
# <a id="Business1"></a> <a href="#top">Back to Top</a>
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
# <a id="Business2"></a> <a href="#top">Back to Top</a>
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
# <a id="Understanding"></a> <a href="#top">Back to Top</a>
#  ## Section 2: Data Understanding
# <a id="Understanding1"></a> <a href="#top">Back to Top</a>
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
# <a id="Understanding2"></a> <a href="#top">Back to Top</a>
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
# <a id="Understanding3"></a> <a href="#top">Back to Top</a>
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


# %%
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
# <a id="Understanding4"></a> <a href="#top">Back to Top</a>
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
#  This bar chart represents income bracket by marital status.
#  Interesting to see a few things, first off the <=50k income bracket highest
#  counts come from the "Never-married" status.  This suggests that marriage does
#  in fact come with alot of financial benefit, as you can see is relevant on the
#  other half of the chart.  As married couples far outmatch any other category
#  counts in the >50k income bracket.  We can confirm this again as most of the
#  divorced, separated, or widowed people are located in the lower income
#  bracket.  Suggesting that, if you want to make over 50k, you might
#  want to get yourself a partner and keep them!
