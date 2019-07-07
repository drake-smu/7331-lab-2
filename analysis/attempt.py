import os
os.chdir('/home/david/7331-lab-2')
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

print(y_train)
