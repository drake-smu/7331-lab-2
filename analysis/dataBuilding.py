
import pandas as pd
import numpy as np
import seaborn as sns
import timeit

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split

# Set Defaults
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
cat_cols = [
    "workclass",
    "marital_status", 
    "occupation",
    "race", 
    "gender",
    "relationship",
    "education",
    "native_country"]

cont_cols = [
    "age", 
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week"]

drop_cols =[]
# drop_cols = [
#     'fnlwgt',
#     "native_country",
#     "education"]

target_col = "target"



def fetch_df():
    """Fetch raw unprocessed data from csv's
    
    Returns:
        dataframe -- returns concatenated dataframe of all data. 
        split how you like later
    """
    df_training = pd.read_csv("data/adult-training.csv",
        names=df_headers, 
        skipinitialspace = True)

    df_test = pd.read_csv("data/adult-test.csv",
        names = df_headers,
        skipinitialspace = True,
        skiprows=1)

    df = pd.concat([df_training,df_test],axis=0)
    # df.info()
    return df

def process_target(df,target_col=target_col):
    if df[target_col] = df["income_bracket"]:
        df[target_col] = (df["income_bracket"]
            .apply(lambda x: ">50K" in x)).astype(int)
    elif df[target_col] = df["gender"]:
        df[target_col] = (df["gender"]
            .apply(lambda x: "Male" in x)).astype(int)
    return df

def process_drops(df, cols):
    return df.drop(cols,axis=1,inplace=True)


####################
## Data Specific
####################

def process_edu(df):
    replace_edu_no = ('1st-4th', '5th-6th','7th-8th','9th', '10th', '11th', '12th', 'Preschool')
    replace_edu_associate = ('Assoc-acdm', 'Assoc-voc')
    replace_edu_diploma = ('Some-college', 'HS-grad')

    df.education = df.education.replace(to_replace=replace_edu_no,value='No Diploma')
    df.education = df.education.replace(to_replace=replace_edu_associate,value='Associates')
    df.education = df.education.replace(to_replace=replace_edu_diploma,value='Diploma')
    return df['education'].str.strip()

def process_native(df):
    # Put countries in their native region continent
    replace_northA = ('United-States', 'Honduras', 'Mexico','Puerto-Rico','Canada', 'Outlying-US(Guam-USVI-etc)', 'Nicaragua', 'Guatemala', 'El-Salvador')
    replace_carib = ('Cuba', 'Jamaica', 'Trinadad&Tobago', 'Haiti', 'Dominican-Republic')
    replace_asia = ('South', 'Cambodia','Thailand','Laos', 'Taiwan', 'China', 'Japan', 'India', 'Iran', 'Philippines', 'Vietnam', 'Hong')
    replace_europe = ('England', 'Germany', 'Portugal', 'Italy', 'Poland', 'France', 'Yugoslavia','Scotland', 'Greece', 'Ireland', 'Hungary', 'Holand-Netherlands')
    replace_sa = ('Columbia', 'Ecuador', 'Peru')
    replace_other = ('?')
    df.native_country = df.native_country.replace(to_replace=replace_northA,value='North America')
    df.native_country = df.native_country.replace(to_replace=replace_northA,value='North America')
    df.native_country = df.native_country.replace(to_replace=replace_carib,value='Caribbean')
    df.native_country = df.native_country.replace(to_replace=replace_asia,value='Asia')
    df.native_country = df.native_country.replace(to_replace=replace_europe,value='Europe') 
    df.native_country = df.native_country.replace(to_replace=replace_sa,value='South America')
    df.native_country = df.native_country.replace(to_replace=replace_other,value='Other')   
    return df

##### END DATA SPECIFIC #########

################################
## Scaling and Encoding Data 
################################

def build_preprocessor(cont_cols,cat_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, cont_cols),
            ('cat', categorical_transformer, cat_cols)])
    return ('preprocessor',preprocessor)


def build_transform(cont_cols,cat_cols):
    return build_preprocessor(cont_cols,cat_cols)

##### END SACALING AND ENCODING #######

def split_df(X,y,split=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    return X_train, X_test, y_train, y_test

def build_df(drops=None):
    df = fetch_df()
    process_target(df, target_col=target_col)
    process_edu(df)
    process_native(df)
    process_drops(df,drops)
    X = df.drop(columns=["income_bracket",target_col])
    y = df[target_col]
    
    return X,y