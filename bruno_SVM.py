#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### Declare helper functions

# In[62]:


# Helper Function to plot confusion Matrix
def plot_confusion_matrix(confusion_matrix, kernel: str, y_limit: list, color_map: str):
    #Plot the confusion Matrix
    fig, ax = plt.subplots(figsize=(10,6))
    title = f'Confusion matrix - SVM - {kernel.upper()}'
    # create heatmap
    sns.heatmap(confusion_matrix, annot = True, cmap = color_map ,fmt='g')
    ax.xaxis.set_label_position("top")
    ax.set_ylim(y_limit)
    ax.xaxis.set_ticklabels(['2','4'])
    ax.yaxis.set_ticklabels(['2','4'])
    plt.title(title, fontsize=20, pad=10.0)
    plt.ylabel('Actual label', fontsize='large')
    plt.xlabel('Predicted label', fontsize='large')
    plt.tight_layout()


# In[63]:


# Train and Test Classifier
# Imput datasets as a list in the folowing format: [X_train,  y_train, X_test, y_test]
def test_classifier(kernel: str, datasets: list):
    
    # Instantiate the classifier
    classifier = SVC(kernel = kernel)
    
    start = time.perf_counter()
    # Fit the classifier to the data
    classifier.fit(datasets[0], datasets[1])
    end = time.perf_counter()
    print('-' * 115)
    print()
    # Print the time taken to train the models
    print(f'{kernel.upper()} processing time: {round((end-start), 6)} s')
    
    # Print Accuracy score on the TRAINING set
    print()
    print('-' * 115)
    print()
    print(f'{kernel.upper()} Training Accuracy score:  {np.round(classifier.score(datasets[0], datasets[1]), 6) * 100}%')
    print()
    # Print Accuracy score on the TEST set
    print(f'{kernel.upper()} Test Accuracy score: {np.round(classifier.score(datasets[2], datasets[3]), 6) * 100}%')   
    print()
    
    # Make pedictions
    y_pred = classifier.predict(datasets[2])
    
    # Get the Confusion Matrix
    cm = confusion_matrix(datasets[-1], y_pred)

    print()
    # Plot the Confusion Matrix
    color_maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    random_index = np.random.randint(0, 4, 1)

    # Call the Confusion Matrix function
    plot_confusion_matrix(cm, kernel, [0, 2], color_maps[random_index[0]])
    
    # Print Classification Report
    print('\t\tClassification Report\n\n',classification_report(datasets[-1], y_pred))


# <div align="center" style='font-size:40px; padding:20px 0px'><strong> EXERCISE 1 </strong></div>

# In[4]:


data_bruno = pd.read_csv('breast_cancer.csv')


# In[5]:


# Get the head of the dataset to have an initial view of its components
data_bruno.head()


# In[6]:


# List of columns
data_bruno.columns


# In[7]:


data_bruno.info()


# In[8]:


#Use a heatmap to visualize missing data
sns.set(rc={"figure.figsize":(10, 6)})
sns.heatmap(data_bruno.isna(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[9]:


data_bruno.isnull().sum().sort_values(ascending=False)


# In[10]:


data_bruno.drop('bare', axis = 1).describe()


# In[11]:


data_bruno['class']


# In[12]:


data_bruno['bare'].replace(to_replace = '?', value = np.nan, inplace = True)


# In[13]:


data_bruno['bare'] = data_bruno['bare'].astype('float')


# In[14]:


data_bruno['bare'].dtype


# In[15]:


data_bruno[data_bruno.columns].mean().round(2)


# In[16]:


data_bruno.fillna(data_bruno.mean(), inplace = True)


# In[17]:


data_bruno.isna().sum()


# In[18]:


data_bruno.drop('ID', axis = 1, inplace = True)


# In[19]:


data_bruno.head()


# In[20]:


proportion = data_bruno['class'].value_counts()/len(data_bruno['class'])


# In[21]:


plt.figure(figsize = (10,6))
sns.barplot(x = [2,4], y = proportion)
plt.xticks(np.arange(2),('Class 2', 'Class 4'))
plt.ylabel('Proportion')
plt.show()


# In[22]:


data_bruno.iloc[:, 1]


# In[23]:


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(projection='3d')
ax.scatter(data_bruno.iloc[:, 1], data_bruno.iloc[:, 2], data_bruno.iloc[:, 3], c=data_bruno['class'], edgecolors='k', cmap='viridis')
ax.view_init(60,70)
plt.show()


# In[24]:


for col in data_bruno.columns:
    sns.displot(data_bruno[col])


# In[25]:


sns.pairplot(data_bruno, hue = 'class')
plt.show()


# In[26]:


sns.heatmap(data_bruno.drop('class', axis = 1).corr(), annot = True, cmap = 'viridis')
plt.show()


# In[27]:


sns.regplot( x = data_bruno['shape'], y = data_bruno['size'])
plt.show()


# In[28]:


X = data_bruno.drop('class', axis = 1)
y = data_bruno['class']


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=98)


# In[30]:


clf_linear_bruno = SVC(kernel="linear", C=0.1)


# In[31]:


clf_linear_bruno.fit(X_train, y_train)


# In[32]:


print(f'Trainning set accuracy: {np.round(clf_linear_bruno.score(X_train, y_train), 6) * 100}%')


# In[33]:


print(f'Test set accuracy: {np.round(clf_linear_bruno.score(X_test, y_test), 6) * 100}%')


# In[34]:


y_pred = clf_linear_bruno.predict(X_test)


# In[35]:


cm = confusion_matrix(y_test, y_pred)


# In[36]:


plot_confusion_matrix(cm, 'Linear', [0,2], 'PuBu')


# In[37]:


print('\t\tClassification Report\n\n',classification_report(y_test, y_pred))


# In[38]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = (lines) => {\n    return false;\n}')


# In[64]:


datasets = [X_train, y_train, X_test, y_test]
kernels = ['rbf', 'poly', 'sigmoid']
for kernel in kernels:
    test_classifier(kernel, datasets)


# <hr></hr>

# <div align="center" style='font-size:40px; padding:20px 0px'><strong> EXERCISE 2 </strong></div>

# In[133]:


data_bruno = pd.read_csv('breast_cancer.csv')


# In[134]:


data_bruno.head()


# In[135]:


data_bruno['bare'].replace(to_replace = '?', value = np.nan, inplace = True)


# In[136]:


data_bruno['bare'] = data_bruno['bare'].astype('float')


# In[137]:


data_bruno['bare'].dtype


# In[138]:


data_bruno.drop('ID', axis = 1, inplace = True)


# In[139]:


X = data_bruno.drop('class', axis = 1)
y = data_bruno['class']


# In[140]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=98)


# In[141]:


num_pipeline_bruno = Pipeline([
('imputer', SimpleImputer(strategy="median")),
('std_scaler', StandardScaler()),
])


# In[142]:


pipe_svm_bruno = Pipeline([
    ('numeric', num_pipeline_bruno),
    ('svm',  SVC(random_state = 98))
])


# In[143]:


num_pipeline_bruno


# In[144]:


param_grid = {'svm__kernel': ['linear', 'rbf', 'poly'],
              'svm__C': [0.1, 0.1, 1, 10, 100],
              'svm__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
              'svm__degree': [2, 3]}


# In[145]:


param_grid


# In[146]:


grid_search_bruno = GridSearchCV(estimator = pipe_svm_bruno,
                                 param_grid = param_grid,
                                 scoring = 'accuracy',
                                 refit = True,
                                 n_jobs = -1,
                                 verbose = 3)


# In[147]:


grid_search_bruno


# In[148]:


start = time.perf_counter()


# In[149]:


grid_search_bruno.fit(X_train, y_train)


# In[150]:


end = time.perf_counter()


# In[151]:


print(f'GridSearchCV processing time: {round((end-start), 2)} s')


# In[153]:


# Best hyperparameters
print("tuned hpyerparameters :(best parameters) ", grid_search_bruno.best_params_)
print("Best Estimator :", grid_search_bruno.best_estimator_)


# In[155]:


num_pipeline_bruno.fit(X_train)


# In[156]:


X_test_transformed = num_pipeline_bruno.transform(X_test)


# In[158]:


pred = grid_search_bruno.predict(X_test)


# In[159]:


grid_search_bruno.score(X_test, y_test)


# In[160]:


best_model_bruno = grid_search_bruno.best_estimator_


# In[161]:


import joblib


# In[162]:


joblib.dump(best_model_bruno, "SVC_model.pkl")


# In[163]:


joblib.dump(pipe_svm_bruno, "full_pipeline.pkl")


# In[164]:


import dill


# In[165]:


dill.dump_session('notebook_env.db')


# <div align="center" style='font-size:40px; padding:20px 0px'><strong> END </strong></div>
