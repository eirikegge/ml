#!/usr/bin/env python
# coding: utf-8

# ***
# # Data Mining, Machine Learning and Deep Learning
# ***

# Helena van Eek (141521) | Magnus Beck Eliassen (141855) | Sabrina Breunig (141706) | Eirik Egge (141164)

# # Information about the Dataset's Features

# `id`: Unique identifier
# 
# `gender`: "Male", "Female" or "Other"
# 
# `age`: Age of the patient
# 
# `hypertension`: "0" if the patient does not have hypertension, "1" if the patient has hypertension
# 
# `heart_disease`: "0" if the patient does not have any heart diseases, "1" if the patient has a heart disease
# 
# `ever_married`: "No" or "Yes"
# 
# `work_type`: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
# 
# `residence_type`: "Rural" or "Urban"
# 
# `avg_glucose_level`: Average glucose level in blood
# 
# `bmi`: Body mass index
# 
# `smoking_status`: "formerly smoked", "never smoked", "smokes" or "Unknow"
# 
# `stroke`: "0" if the patient does not have a stroke, "1" if the patient has a stroke

# # Exploratory Data Analysis

# In[3]:


# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# Read the dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')


# In[5]:


# Look at the head of the dataset
df.head()


# In[6]:


# Get a summary of the dataset
df.describe()


# In[7]:


# Get information about the types of the features
df.info()


# In[8]:


# Get dimensions of the dataset
print('Number of rows:', df.shape[0])
print('Number of columns:', df.shape[1])


# In[9]:


# Get an overview of the unique values for each feature
print("Unique Values per Variable")
for col in df.columns:
    un=df[col].unique()
    print("\n\nUnique Values in {}:\n{}".format(col,un))


# In[10]:


# Get the absolute count of each unique value for each feature
df_col=['gender','age',	'hypertension',	'heart_disease',	'ever_married',	'work_type',	'Residence_type',	'avg_glucose_level',	'bmi',	'smoking_status',	'stroke']

for i in df_col:
    print(df[i].value_counts())


# In[11]:


# Get the relative share of each unique value for each feature
pd.options.display.float_format = '{:.2%}'.format
print(df.groupby('hypertension')['hypertension'].count().apply(lambda x: x/ len(df)))
print(df.groupby('heart_disease')['heart_disease'].count().apply(lambda x: x/ len(df)))
print(df.groupby('stroke')['stroke'].count().apply(lambda x: x/ len(df)))
print(df.groupby('gender')['gender'].count().apply(lambda x: x/ len(df)))
print(df.groupby('ever_married')['ever_married'].count().apply(lambda x: x/ len(df)))
print(df.groupby('work_type')['work_type'].count().apply(lambda x: x/ len(df)))
print(df.groupby('smoking_status')['smoking_status'].count().apply(lambda x: x/ len(df)))
print(df.groupby('Residence_type')['Residence_type'].count().apply(lambda x: x/ len(df)))

pd.reset_option('display.float_format')


# In[12]:


# Check for unique id
lst = list(df['id'].unique())
len(lst)
# Every ID is unique in this dataset

# Drop id column as it does not add any value for our purposes
df.drop(columns=['id'], inplace=True)


# In[13]:


# Test how many of the "Unknown" smokers are under the age of 21
df_test_smoking = df[df['age'] <= 21]

df_test_smoking.smoking_status.value_counts()


# In[14]:


# Compare to the total smoker feature distribution
df.smoking_status.value_counts()

# About 50% of the "Unknown" smoker are below the age of 21


# In[15]:


#  Drop "Other" from Gender as it only appears once and fix index
df = df[df["gender"].str.contains("Other")==False].reset_index(drop = True)


# In[16]:


# Convert income to binary target variable
df['gender'].replace({'Female':0,'Male':1}, inplace=True)
df['ever_married'].replace({'No':0,'Yes':1}, inplace=True)
df['Residence_type'].replace({'Rural':0,'Urban':1}, inplace=True)


# ### Target Variable

# In[17]:


# Display the count of the target
print(df['stroke'].value_counts())

#Plot the count of the target
ncount = len(df['stroke'])
ax = sns.countplot(x=df['stroke'], palette="pastel")
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom')

# The dataset is very imbalanced, 95% did not have a stroke


# ### Feature Variables

# In[18]:


# Separate numeric and categorical columns
df_cat = df.select_dtypes(include=['object']).copy()
df_num = df.select_dtypes(include=["int64", 'float64']).copy()


# In[19]:


# Plot the histograms for the numerical features
fig, axs = plt.subplots(2, 4, figsize=(30,10))

# Plot the countplots for the categorical features
sns.countplot(x = df["hypertension"], ax=axs[0, 0], palette="pastel")
sns.countplot(x = df["heart_disease"], ax=axs[0, 1], palette="pastel")
sns.countplot(x = df["ever_married"], ax=axs[0, 2], palette="pastel")
sns.countplot(x = df["gender"], ax=axs[0, 3], palette="pastel")
sns.countplot(x = df["Residence_type"], ax=axs[1, 0], palette="pastel")
sns.countplot(x = df["work_type"], ax=axs[1, 1], palette="pastel")
sns.countplot(x = df["smoking_status"], ax=axs[1, 2], palette="pastel")
sns.countplot(x = df["stroke"], ax=axs[1, 3], palette="pastel")

plt.show()


# In[38]:


# Visualise the spread of smokers for each gender
gs = fig.add_gridspec(3,3)
gs.update(wspace=0.4, hspace=0.4)
gs.update(wspace=0.4, hspace=0.4)
cat1 = sns.catplot(x = "gender", y = "stroke", hue = "smoking_status", kind = "bar", palette="pastel", data = df)
cat2 = sns.catplot(x = "gender", y = "stroke", hue = "work_type",  kind = "bar", palette="pastel", data = df)
cat3 = sns.catplot(x = "gender", y = "stroke", hue = "heart_disease", kind = "bar", palette="pastel", data = df)
cat4 = sns.catplot(x = "gender", y = "stroke", hue = "hypertension", kind = "bar", palette="pastel", data = df)
cat5 = sns.catplot(x = "gender", y = "stroke", hue = "ever_married", kind = "bar", palette="pastel", data = df)


# In[ ]:


fig = plt.figure(figsize=(50,10))
gs = fig.add_gridspec(3,3)
gs.update(wspace=0.4, hspace=0.4)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[2,0])
kde1 = sns.kdeplot(ax=ax0,x=df.loc[df['stroke']==1]['avg_glucose_level'],label='Stroke',shade=True, palette= "pastel", data =df_num)
kde2 = sns.kdeplot(ax=ax0,x=df.loc[df['stroke']==0]['avg_glucose_level'],label='No Stroke',shade=True, palette= "pastel", data=df_num)
kde3 = sns.kdeplot(ax=ax1,x=df.loc[df['stroke']==1]['bmi'],label='Stroke',shade=True, palette= "pastel", legend= True, data=df_num)
kde4 = sns.kdeplot(ax=ax1,x=df.loc[df['stroke']==0]['bmi'],label='No Stroke',shade=True, palette= "pastel", data=df_num)
kde5 = sns.kdeplot(ax=ax2,x=df.loc[df['stroke']==1]['age'],label='Stroke',shade=True, palette= "pastel", data=df_num)
kde6 = sns.kdeplot(ax=ax2,x=df.loc[df['stroke']==0]['age'],label='No Stroke',shade=True, palette= "pastel", data=df_num)

l1 = ax0.legend(loc = 'upper right')
l2 = ax1.legend(loc = 'upper right')
l3 = ax2.legend(loc = 'upper right')


# # Data Pre-Processing

# ### Handling Missing Values

# In[20]:


# Detect missing values
df.isna().sum()


# In[21]:


# We need to handle the missing values for bmi
# In the EDA, we saw that bmi is close to normal distribution but has some outliers on the right-skewed part
# Replace missing values with median
df_num['bmi'].fillna(df_num['bmi'].median(), inplace=True)
df_num.isnull().sum()


# ### Encoding Categorical Variables

# In[22]:


from sklearn.preprocessing import OneHotEncoder

# Encode all categorical variables with the OneHotEncoder
encoder_ohe = OneHotEncoder(sparse = False)

df_cat_enc = pd.DataFrame(encoder_ohe.fit_transform(df_cat), columns = ('Govt_job',
       'never_worked', 'private_job', 'self-employed', 'children', 'smokes_unknown', 'formerly_smoked',
       'never_smoked', 'smoker'))

df_new = pd.concat([df_num, df_cat_enc], axis=1)

df_new.shape


# In[23]:


# Plot the correlation of encoded dataframe
corr = df_new.corr().round(3)
plt.figure(figsize=(20,15))
ht = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)


# ### Outlier Detection &amp; Removal

# In[24]:


# Boxplot the continous features to visualize outliers
df_continues = df_new[["age", "bmi", "avg_glucose_level"]]
boxplot = df_continues.boxplot()


# In[25]:


# Remove the outliers based on the IQR score for the continous features
Q1 = df_continues.quantile(0.10)
Q3 = df_continues.quantile(0.90)
IQR = Q3 - Q1
j = Q1 - 1.5 * IQR
k = Q3 + 1.5 * IQR

def IQR_score(data):
    count = 0
    for col in df_continues.columns:
        for i in data[col]:
            if i < j[col] or i > k[col]:
                count += 1
                data = data[data[col]!=i]
    return pd.DataFrame(data)

df_enc = IQR_score(df_new)
df_enc.shape


# In[26]:


# Boxplot the continous features after outlier removal
boxplot = df_enc[['age','bmi','avg_glucose_level']].boxplot()


# In[27]:


df_enc.shape


# ### Checking for  Multicollinearity ( VIF = 1 → No correlation // VIF = 1 to 5 → Moderate correlation // VIF &gt;10 → High correlation)

# In[28]:


#Checking for multicolinearity - "10" as the maximum level of VIF (Hair et al., 1995)
#! pip install statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_scores(X):
    VIF_Scores = pd.DataFrame()
    VIF_Scores["Independent Features"] = X.columns
    VIF_Scores["VIF Scores"] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
    return VIF_Scores

df1 = df_enc.iloc[:,:-1]
vif_scores(df1)


# In[29]:


# Drop id column as it does not add any value for our purposes
df_enc.drop(columns=['private_job'], inplace=True)

df2 = df_enc.iloc[:,:-1]
vif_scores(df2)


# ### Split the dataset

# In[30]:


from sklearn.model_selection import train_test_split

# Set features as input variable X
X = df_enc.drop(['stroke'], axis = 1)

# Set "stroke" as target variable y
y = df_enc.stroke

# Use a 75/25 split for the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Get column names of X_train
col_names = list(X_train.columns.values)


# ### Pipelines for Scaling

# In[31]:


from sklearn.compose import ColumnTransformer
from sklearn import preprocessing

# Select the numerical float features
numeric_features = ['age', 'bmi','avg_glucose_level']

# Preprocessing with StandardScaler
numeric_transformer_s = preprocessing.StandardScaler()
preprocessor_s = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_s, numeric_features)])

# Preprocessing with MinMaxScaler
numeric_transformer_mm = preprocessing.MinMaxScaler()
preprocessor_mm = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_mm, numeric_features)])


# # Baseline Model Performance

# In[32]:


# Import packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# ### Random Forest Baseline Model

# In[33]:


start = time.time()

# Create a Random Forest Classifier
rfc_pipeline = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_rf_bl', RandomForestClassifier(random_state = 42,))
])

# Train the model using the training set
rfc_pipeline.fit(X_train, y_train)

# Predict the response for test dataset
y_pred_rf_bl = rfc_pipeline.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for RF Classifier
disp = plot_confusion_matrix(rfc_pipeline, X_test, y_test, cmap=plt.cm.Blues,normalize= 'true')
plt.title("Confusion Matrix RF")
plt.show()

# Classification report for RF Classifier
print(classification_report(y_test, y_pred_rf_bl))


# ### Neural Network Baseline Model

# In[34]:


start = time.time()

# Create a MLP Classifier
mlp_pipeline = Pipeline(steps = [
    (('preprocessor', preprocessor_mm)),
    ('clf_mlp_bl', MLPClassifier(random_state = 42,))
])

# Train the model using the training set
mlp_pipeline.fit(X_train,y_train)

# Predict the response for test dataset
y_pred_nn_bl = mlp_pipeline.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for MLP Classifier
disp = plot_confusion_matrix(mlp_pipeline, X_test, y_test, cmap=plt.cm.Blues,normalize= 'true')
plt.title("Confusion Matrix MLP")
plt.show()

# Classification report for MLP Classifier
print(classification_report(y_test, y_pred_nn_bl))


# ### SVM Baseline Model

# In[35]:


start = time.time()

# Create a SVM Classifier
svm_pipeline = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_svm_bl', SVC(random_state = 42,))
])

# Train the model using the training set
svm_pipeline.fit(X_train,y_train)

# Predict the response for test dataset
y_pred_svm_bl = svm_pipeline.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for SVM Classifier
disp = plot_confusion_matrix(svm_pipeline, X_test, y_test, cmap=plt.cm.Blues,normalize= 'true')
plt.title("Confusion Matrix SVM")
plt.show()

# Classification report for SVM Classifier
print(classification_report(y_test, y_pred_svm_bl))


# # Balance the Data (SMOTE and ADASYN)

# In[36]:


get_ipython().system(' pip install imblearn')


# ### SMOTE

# In[37]:


from imblearn.over_sampling import SMOTE

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

start = time.time()

# Apply over-sampling method SMOTE
sm = SMOTE(random_state = 42)
X_train=np.array(X_train)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train.values.ravel())

# Create dataframe for oversampled training data
X_train_sm_df = pd.DataFrame(X_train_sm, columns = col_names)

end = time.time()
print(round(end - start,2), "seconds running time\n")

print("After OverSampling, counts of label '1': {}".format(sum(y_train_sm==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_sm==0)))

# Data after oversampling
cnt = sns.countplot(x = y_train_sm, data = df_enc, palette="pastel")


# ### ADASYN

# In[38]:


from imblearn.over_sampling import ADASYN

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

start = time.time()

# Apply over-sampling method ADASYN
ada = ADASYN(random_state = 42)
X_train_ada, y_train_ada = ada.fit_resample(X_train, y_train)

# Create dataframe for oversampled training data
X_train_ada_df = pd.DataFrame(X_train_ada, columns = col_names)

end = time.time()
print(round(end - start,2), "seconds running time\n")

print("After OverSampling, counts of label '1': {}".format(sum(y_train_ada==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_ada==0)))

# Data after oversampling
cnt = sns.countplot(x = y_train_ada, data = df_enc, palette="pastel")


# ### Random Forest Baseline Model with SMOTE

# In[39]:


start = time.time()

# Create a Random Forest Classifier
rfc_pipeline_bl_sm = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_rf_bl_sm', RandomForestClassifier(random_state = 42))
])

# Train the model using the training set
rfc_pipeline_bl_sm.fit(X_train_sm_df, y_train_sm)

# Predict the response for test dataset
y_pred_rf_bl_sm = rfc_pipeline_bl_sm.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for RF Classifier
disp = plot_confusion_matrix(rfc_pipeline_bl_sm, X_test, y_test, cmap=plt.cm.Blues,normalize= 'true')
plt.title("Confusion Matrix RF (SMOTE)")
plt.show()

# Classification report for RF Classifier
print(classification_report(y_test, y_pred_rf_bl_sm))


# ### Random Forest Baseline Model with ADASYN

# In[40]:


start = time.time()

# Create a Random Forest Classifier
rfc_pipeline_bl_ada = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_rf_bl_ada', RandomForestClassifier(random_state = 42))
])

# Train the model using the training set
rfc_pipeline_bl_ada.fit(X_train_ada_df, y_train_ada)

# Predict the response for test dataset
y_pred_rf_bl_ada = rfc_pipeline_bl_ada.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for RF Classifier
disp = plot_confusion_matrix(rfc_pipeline_bl_ada, X_test, y_test, cmap=plt.cm.Blues,normalize= 'true')
plt.title("Confusion Matrix RF (ADASYN)")
plt.show()

# Classification report for RF Classifier
print(classification_report(y_test, y_pred_rf_bl_ada))


# ### Neural Network Baseline Model with SMOTE

# In[41]:


start = time.time()

# Create a MLP Classifier
mlp_pipeline_bl_sm = Pipeline(steps = [
    (('preprocessor', preprocessor_mm)),
    ('clf_mlp_bl_sm', MLPClassifier(random_state = 42))
])

# Train the model using the training set
mlp_pipeline_bl_sm.fit(X_train_sm_df, y_train_sm)

# Predict the response for test dataset
y_pred_mlp_bl_sm = mlp_pipeline_bl_sm.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for MLP Classifier
disp = plot_confusion_matrix(mlp_pipeline_bl_sm, X_test, y_test, cmap=plt.cm.Blues,normalize= 'true')
plt.title("Confusion Matrix MLP (SMOTE)")
plt.show()

# Classification report for MLP Classifier
print(classification_report(y_test, y_pred_mlp_bl_sm))


# ### Neural Network Baseline Model with ADASYN

# In[42]:


start = time.time()

# Create a MLP Classifier
mlp_pipeline_bl_ada = Pipeline(steps = [
    (('preprocessor', preprocessor_mm)),
    ('clf_mlp_bl_ada', MLPClassifier(random_state = 42))
])

# Train the model using the training set
mlp_pipeline_bl_ada.fit(X_train_ada_df, y_train_ada)

# Predict the response for test dataset
y_pred_mlp_bl_ada = mlp_pipeline_bl_ada.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for MLP Classifier
disp = plot_confusion_matrix(mlp_pipeline_bl_ada, X_test, y_test, cmap=plt.cm.Blues,normalize= 'true')
plt.title("Confusion Matrix MLP (ADASYN)")
plt.show()

# Classification report for MLP Classifier
print(classification_report(y_test, y_pred_mlp_bl_ada))


# ### SVM Baseline Model with SMOTE

# In[43]:


start = time.time()

# Create a SVM Classifier
svm_pipeline_bl_sm = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_svm_bl_sm', SVC(random_state = 42))
])

# Train the model using the training set
svm_pipeline_bl_sm.fit(X_train_sm_df, y_train_sm)

# Predict the response for test dataset
y_pred_svm_bl_sm = svm_pipeline_bl_sm.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for SVM Classifier
disp = plot_confusion_matrix(svm_pipeline_bl_sm, X_test, y_test, cmap=plt.cm.Blues,normalize= 'true')
plt.title("Confusion Matrix SVM (SMOTE)")
plt.show()

# Classification report for SVM Classifier
print(classification_report(y_test, y_pred_svm_bl_sm))


# ### SVM Baseline Model with ADASYN

# In[44]:


start = time.time()

# Create a SVM Classifier
svm_pipeline_bl_ada = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_svm_bl_ada', SVC(random_state = 42))
])

# Train the model using the training set
svm_pipeline_bl_ada.fit(X_train_ada_df, y_train_ada)

# Predict the response for test dataset
y_pred_svm_bl_ada = svm_pipeline_bl_ada.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for SVM Classifier
disp = plot_confusion_matrix(svm_pipeline_bl_ada, X_test, y_test, cmap=plt.cm.Blues,normalize= 'true')
plt.title("Confusion Matrix SVM (ADASYN)")
plt.show()

# Classification report for SVM Classifier
print(classification_report(y_test, y_pred_svm_bl_ada))


# # Performance of Balanced & Tuned Models

# ### Gridsearch Function for class-wise optimization

# In[45]:


# Define a gridsearch function for class-wise performance optimization
def grid_search_wrapper(clf, X_train, y_train, param_grid):
    """
    fits a GridSearchCV classifier using the f-beta score for recall optimization
    """
    scoring = make_scorer(fbeta_score, beta = 2, pos_label = 1)
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf, param_grid, scoring= scoring, 
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Make the predictions
    y_pred = grid_search.predict(X_test)

    print('Best params for {}'.format(scoring))
    print(grid_search.best_params_)

    # Confusion matrix on the test data
    print('\nConfusion matrix of Classifier optimized for {} on the test data:'.format(scoring))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search


# ### Fine Tuning Random Forest

# In[46]:


# Set parameters for Gridsearch
param_grid_rf = {
    'clf_rf__min_samples_split': [1, 3, 5], 
    'clf_rf__n_estimators' : [10, 50, 100],
    'clf_rf__max_depth': [1, 3, 5, 10],
    'clf_rf__max_features': ["auto", "sqrt", "log2", "None"],
    'clf_rf__ccp_alpha':[0.01,0.05,0.1,0.5]
    }

# Set pipeline for Gridsearch
rfc_pipeline = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_rf', RandomForestClassifier(random_state=42,class_weight='balanced'))
])


# In[47]:


grid_search_wrapper(rfc_pipeline, X_train_sm_df, y_train_sm, param_grid_rf)


# In[48]:


grid_search_wrapper(rfc_pipeline, X_train_ada_df, y_train_ada, param_grid_rf)


# 
# ### Random Forest with SMOTE and Tuned Hyperparameters

# In[49]:


start = time.time()

# Create a Random Forest Classifier with tuned hyperparameters
rfc_pipeline_tuned_sm = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_rf', RandomForestClassifier(random_state = 42, ccp_alpha = 0.05, max_depth = 1, max_features = 'auto', min_samples_split = 3, n_estimators = 10, class_weight='balanced'))
])

# Train the model using the training sets
rfc_pipeline_tuned_sm.fit(X_train_sm_df, y_train_sm)

# Predict the response for test dataset
y_pred_rf_sm = rfc_pipeline_tuned_sm.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for tuned RF Classifier
disp = plot_confusion_matrix(rfc_pipeline_tuned_sm, X_test, y_test, cmap=plt.cm.Blues,normalize= 'true')
plt.title("Confusion Matrix RF Tuned (SMOTE)")
plt.show()

# Classification report for tuned RF Classifier
print(classification_report(y_test, y_pred_rf_sm))


# ### Random Forest with ADASYN and Tuned Hyperparameters

# In[50]:


start = time.time()

# Create a Random Forest Classifier with tuned hyperparameters
rfc_pipeline_tuned_ada = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_rf', RandomForestClassifier(random_state = 42, ccp_alpha = 0.05, max_depth = 1, max_features = 'auto', min_samples_split = 3, n_estimators = 10, class_weight='balanced'))
])

# Train the model using the training set
rfc_pipeline_tuned_ada.fit(X_train_ada_df, y_train_ada)

# Predict the response for test dataset
y_pred_rf_ada = rfc_pipeline_tuned_ada.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for tuned RF Classifier
disp = plot_confusion_matrix(rfc_pipeline_tuned_ada, X_test, y_test, cmap=plt.cm.Blues, normalize= 'true')
plt.title("Confusion Matrix RF Tuned (ADASYN)")
plt.show()

# Classification report for tuned RF Classifier
print(classification_report(y_test, y_pred_rf_ada))


# ### Fine Tuning Neural Network

# In[51]:


# Set parameters for Gridsearch
param_grid_mlp = {
    'clf_mlp__hidden_layer_sizes': [(20,20,20), (30,30,30), (50,50,50), (100,)],
    'clf_mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'clf_mlp__solver': ['lbfgs', 'sgd', 'adam'],
    'clf_mlp__alpha': [0.0001,0.025, 0.05],
    'clf_mlp__learning_rate': ['constant','adaptive', 'invscaling'],
}

# Set pipeline for Gridsearch
mlp_pipeline = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_mlp', MLPClassifier(random_state = 42))
    ])


# In[52]:


grid_search_wrapper(mlp_pipeline, X_train_sm_df, y_train_sm, param_grid_mlp)


# In[53]:


grid_search_wrapper(mlp_pipeline, X_train_ada_df, y_train_ada, param_grid_mlp)


# ### Neural Network with SMOTE and Tuned Hyperparameters

# In[54]:


start = time.time()

# Create a Neural Network Classifier with tuned hyperparameters
mlp_pipeline_tuned_sm = Pipeline(steps = [
    (('preprocessor', preprocessor_mm)),
    ('clf_mlp', MLPClassifier(hidden_layer_sizes=(50,50,50), activation='tanh', solver='lbfgs', alpha = 0.05, learning_rate = 'constant', random_state =42))
])

# Train the model using the training set
mlp_pipeline_tuned_sm.fit(X_train_sm_df, y_train_sm)

# Predict the response for test dataset
y_pred_nn_sm = mlp_pipeline_tuned_sm.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for tuned MLP Classifier
disp = plot_confusion_matrix(mlp_pipeline_tuned_sm, X_test, y_test, cmap=plt.cm.Blues, normalize= 'true')
plt.title("Confusion Matrix MLP Tuned (SMOTE)")
plt.show()

# Classification report for tuned MLP Classifier
print(classification_report(y_test,y_pred_nn_sm))


# ### Neural Network with ADASYN and Tuned Hyperparameters

# In[73]:


start = time.time()

# Create a Neural Network Classifier with tuned hyperparameters
mlp_pipeline_tuned_ada = Pipeline(steps = [
    (('preprocessor', preprocessor_mm)),
    ('clf_mlp', MLPClassifier(hidden_layer_sizes=(50,50,50), activation='relu', solver='adam', alpha = 0.025, learning_rate = 'constant', random_state =42))
])

# Train the model using the training set
mlp_pipeline_tuned_ada.fit(X_train_ada_df,y_train_ada)

# Predict the response for test dataset
y_pred_nn_ada = mlp_pipeline_tuned_ada.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for tuned MLP Classifier
disp = plot_confusion_matrix(mlp_pipeline_tuned_ada, X_test, y_test, cmap=plt.cm.Blues, normalize= 'true')
plt.title("Confusion Matrix MLP Tuned (ADASYN)")
plt.show()

# Classification report for tuned MLP Classifier
print(classification_report(y_test,y_pred_nn_ada))


# ### Fine Tuning SVM

# In[56]:


# Set parameters for Gridsearch
param_grid_svc = {
        'clf_svm__C' : [0.1, 1, 10, 100, 1000],
        'clf_svm__gamma' : ['scale', 'auto'],
        'clf_svm__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 
        'clf_svm__degree' : [0, 2, 4, 6],
        'clf_svm__max_iter': [750, 1000]
        }

# Set pipeline for Gridsearch
svm_pipeline = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_svm', SVC(random_state=42, class_weight = 'balanced'))
])


# In[57]:


grid_search_wrapper(svm_pipeline, X_train_sm_df, y_train_sm, param_grid_svc)


# In[58]:


grid_search_wrapper(svm_pipeline, X_train_ada_df, y_train_ada, param_grid_svc)


# ### SVM with SMOTE and Tuned Hyperparameters

# In[59]:


start = time.time()

# Create a SVM Classifier with tuned hyperparameters
svm_pipeline_tuned_sm = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_svm', SVC(random_state=42, C = 0.1, degree = 0, gamma = 'scale', kernel = 'linear', max_iter = 750, class_weight='balanced'))
])

# Train the model using the training set
svm_pipeline_tuned_sm.fit(X_train_sm_df, y_train_sm)

# Predict the response for test dataset
y_pred_svm_sm = svm_pipeline_tuned_sm.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for tuned SVM Classifier
disp = plot_confusion_matrix(svm_pipeline_tuned_sm, X_test, y_test, cmap=plt.cm.Blues, normalize= 'true')
plt.title("Confusion Matrix SVM Tuned (SMOTE)")
plt.show()

# Classification report for tuned SVM Classifier
print(classification_report(y_test,y_pred_svm_sm))


# ### SVM with ADASYN and Tuned Hyperparameters

# In[60]:


start = time.time()

# Create a SVM Classifier with tuned hyperparameters
svm_pipeline_tuned_ada = Pipeline(steps = [
    (('preprocessor', preprocessor_s)),
    ('clf_svm', SVC(class_weight='balanced', random_state = 42, C = 0.1, degree = 0, gamma = 'scale',kernel = 'linear', max_iter = 750))
])

# Train the model using the training set
svm_pipeline_tuned_ada.fit(X_train_ada_df, y_train_ada)

# Predict the response for test dataset
y_pred_svm_ada = svm_pipeline_tuned_ada.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for tuned SVM Classifier
disp = plot_confusion_matrix(svm_pipeline_tuned_ada, X_test, y_test, cmap=plt.cm.Blues, normalize= 'true')
plt.title("Confusion Matrix SVM Tuned (ADASYN)")
plt.show()

# Classification report for tuned SVM Classifier
print(classification_report(y_test,y_pred_svm_ada))


# ### Voting Classifier (SMOTE)

# In[61]:


start = time.time()

# Create a Hard Voting Classifier with tuned classifiers
eclf_sm = VotingClassifier(estimators=[('svm', svm_pipeline_tuned_sm), ('rf', rfc_pipeline_tuned_sm), ('mlp', mlp_pipeline_tuned_sm)], voting='hard')

# Train the model using the training set
eclf_sm.fit(X_train_sm_df, y_train_sm)

# Predict the response for test dataset
y_pred_vot_sm = eclf_sm.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for Voting Classifier
disp = plot_confusion_matrix(eclf_sm, X_test, y_test, cmap=plt.cm.Blues, normalize= 'true')
plt.title("Confusion Matrix Voting Classifier (SMOTE)")
plt.show()

# Classification report for Voting Classifier
print(classification_report(y_test,y_pred_vot_sm))


# ### Voting Classifier (ADASYN)

# In[62]:


start = time.time()

# Create a Hard Voting Classifier with tuned classifiers
eclf_ada = VotingClassifier(estimators=[('svm', svm_pipeline_tuned_ada), ('rf', rfc_pipeline_tuned_ada), ('mlp', mlp_pipeline_tuned_ada)], voting='hard')

# Train the model using the training set
eclf_ada.fit(X_train_ada_df, y_train_ada)

# Predict the response for test dataset
y_pred_vot_ada = eclf_ada.predict(X_test)

end = time.time()
print(round(end - start,2), "seconds running time")

# Confusion matrix for Voting Classifier
disp = plot_confusion_matrix(eclf_ada, X_test, y_test, cmap=plt.cm.Blues, normalize= 'true')
plt.title("Confusion Matrix Voting Classifier (ADASYN)")
plt.show()

# Classification report for Voting Classifier
print(classification_report(y_test,y_pred_vot_ada))


# ### ROC Curves (SMOTE)

# In[63]:


# Create lists with tuned and SMOTE balanced Classifiers
classifer_list_sm = [svm_pipeline_tuned_sm.fit(X_train_sm_df, y_train_sm), rfc_pipeline_tuned_sm.fit(X_train_sm_df, y_train_sm), mlp_pipeline_tuned_sm.fit(X_train_sm_df, y_train_sm)]

# Create lists with tuned and ADASYN balanced Classifiers
classifier_list_ada = [svm_pipeline_tuned_ada.fit(X_train_ada_df, y_train_ada), rfc_pipeline_tuned_ada.fit(X_train_ada_df, y_train_ada), mlp_pipeline_tuned_ada.fit(X_train_ada_df, y_train_ada)]

# By default, estimators.classes_[1] is considered as the positive class.
lst = ['SVM','RFC','MLP']
ax = plt.gca()
for idx,val in enumerate(classifer_list_sm):
    plot_roc_curve(val,X_test,y_test,ax=ax,name=(lst[idx]))


# ### ROC Curves (ADASYN)

# In[64]:


# Create ROC curves for classifiers with ADASYN
ax1= plt.gca()
for idx,val in enumerate(classifier_list_ada):
    plot_roc_curve(val,X_test,y_test,ax=ax1,name=lst[idx])


# ### Comparison of all ROC Curves

# In[65]:


# ROC curves for all classifiers (incl. DC)
lst = ['SVM_sm','RFC_sm','MLP_sm']
lst1 = ['SVM_ada','RFC_ada','MLP_ada']
plt.figure()
ax = plt.gca()
for idx,val in enumerate(classifer_list_sm):
    plot_roc_curve(val,X_test,y_test,ax=ax,name=(lst[idx]))
 
for idx,val in enumerate(classifier_list_ada):
    plot_roc_curve(val,X_test,y_test,ax=ax,name=lst1[idx])

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
        label='DC (AUC = 0.50)', alpha=.8)

ax.legend(loc = 'lower right')

plt.savefig('all_roc_curves_w_dc',dpi =300, edgecolor = 'w', facecolor = 'w')


# # Feature Importance Comparison

# In[66]:


from sklearn.inspection import permutation_importance

# Set scoring for permutation
scoring = make_scorer(fbeta_score, beta = 2, pos_label = 1)


# ### Feature Importance for Random Forest Classifier (SMOTE)

# In[67]:


feat_importances = pd.Series(rfc_pipeline_tuned_sm[1].fit(X_train_sm_df, y_train_sm).feature_importances_, index = X.columns)
res = feat_importances.nlargest(5)
print(res)

feat = res
# Plot features in a horizontal bar chart
feat.sort_values(inplace = True)
feat.plot(kind='barh')
plt.title("Most important features (RF)")
plt.show()


# ### Feature Importance for Random Forest Classifier (ADASYN)

# In[68]:


feat_importances = pd.Series(rfc_pipeline_tuned_ada[1].fit(X_train_ada_df, y_train_ada).feature_importances_, index = X.columns)
res = feat_importances.nlargest(5)
print(res)

feat = res
# Plot features in a horizontal bar chart
feat.sort_values(inplace = True)
feat.plot(kind='barh')
plt.title("Most important features (RF)")
plt.show()


# ### Feature Importance Neural Network (SMOTE)

# In[69]:


#Permutation Importance is a method that provides a way to compute feature importances for any black-box estimator by measuring how score decreases when a feature is not available", 
#The Multi-Layer Perceptron does not have an intrinsic feature importance, such as Decision Trees and Random Forests do. Neural Networks rely on complex co-adaptations of weights during the training phase instead of measuring and comparing quality of splits.

# Perform permutation importance according to roc_auc scoring
results = permutation_importance(mlp_pipeline_tuned_sm.fit(X_train_sm_df, y_train_sm), X_train_sm_df, y_train_sm, scoring=scoring, random_state=42)
# Get importance
importance = results.importances_mean

feat_importances_mlp = pd.Series(importance, index = X_train_sm_df.columns)
# Get the top k most important features and their coffiecients
res = feat_importances_mlp.nlargest(5)
# Return result
print(res)

feat = res
# Plot features in a horizontal bar chart
feat.sort_values(inplace = True)
feat.plot(kind='barh')
plt.title("Most important features (MLP)")
plt.show()


# ### Feature Importance Neural Network (ADASYN)

# In[70]:


# Perform permutation importance according to roc_auc scoring
results = permutation_importance(mlp_pipeline_tuned_ada.fit(X_train_ada_df, y_train_ada), X_train_ada_df, y_train_ada, scoring=scoring, random_state=42)
# Get importance
importance = results.importances_mean

feat_importances_mlp = pd.Series(importance, index = X_train_ada_df.columns)
# Get the top k most important features and their coffiecients
res = feat_importances_mlp.nlargest(5)
# Return result
print(res)

feat = res
# Plot features in a horizontal bar chart
feat.sort_values(inplace = True)
feat.plot(kind='barh')
plt.title("Most important features (MLP)")
plt.show()


# ### Feature Importance for SVM (SMOTE)

# In[71]:


feat_importances_svm_sm = pd.Series(svm_pipeline_tuned_sm[1].fit(X_train_sm_df, y_train_sm).coef_[0],index = X.columns)
res = feat_importances_svm_sm.nlargest(5)
print(res)

feat = res
# Plot features in a horizontal bar chart
feat.sort_values(inplace = True)
feat.plot(kind='barh')
plt.title("Most important features (SVM)")
plt.show()


# ### Feature Importance SVM (ADASYN)

# In[72]:


feat_importances_svm_sm = pd.Series(svm_pipeline_tuned_ada[1].fit(X_train_ada_df, y_train_ada).coef_[0],index = X.columns)
res = feat_importances_svm_sm.nlargest(5)
print(res)

feat = res
# Plot features in a horizontal bar chart
feat.sort_values(inplace = True)
feat.plot(kind='barh')
plt.title("Most important features (SVM)")
plt.show()

