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
#  ## Table of Contents
#  We'll have to put one in tomorrow, i'm just going to start writing
#%% [markdown]
#  supposed to be a table on contents, but i dont know how to make
#  markdown do that yet.
#%% [markdown]
#  ## Section 1: Business Understanding
#  ### Section 1a: Describe the purpose of the data set you selected.
#  We chose this dataset from the UCI's machine learning repository for its categorical
#  predictive attributes.  It contains 1994 Census data pulled from the US Census
#  database.  The prediction task we've set forth is to predict if a persons
#  salary range is >50k in a 1994, based on the various categorical/numerical
#  attributes in the census database. The link to the data source is below:
# 
#  https://archive.ics.uci.edu/ml/datasets/census+income
# 
#  ### Section 1b: Describe how you would define and measure the outcomes from the dataset.
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
#  ## Section 2: Data Understanding
#  ### Section 2a: Describe the meaning and type of data for each attribute
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
#  ### Section 2b: Data Quality
#  Verify data quality: Explain any missing values, duplicate data, and outliers.
#  Are those mistakes? How do we deal with these problems?
# 
#  In the next code section we will import our libraries and data, then begin looking at
#  missing data, duplicate data, and outliers.
