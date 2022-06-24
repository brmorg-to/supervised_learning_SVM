#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import umap


# ### Declare helper functions
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

# Load the dataset
data_bruno = pd.read_csv('breast_cancer.csv')

# Get the head of the dataset to have an initial view of its components
data_bruno.head()

# List of columns
data_bruno.columns

# Get information about the structure and types of the dataset
data_bruno.info()

#Use a heatmap to visualize missing data
sns.set(rc={"figure.figsize":(12, 7)})
sns.heatmap(data_bruno.isna(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# `No missing data visible in the heatmap`

# Confirm the visual cue that the dataset is integral and that there are no missing data
data_bruno.isnull().sum().sort_values(ascending=False)

# Statistics
data_bruno.describe()


# A glimpse inside the target variables vector
data_bruno['class']

# Remove '?' from 'bare' features column
data_bruno['bare'].replace(to_replace = '?', value = np.nan, inplace = True)

# Convert 'bare' column into float
data_bruno['bare'] = data_bruno['bare'].astype('float')

# Confirm the change to numeric
data_bruno['bare'].dtype

# Check the mean of each column in the dataframe
data_bruno[data_bruno.columns].mean().round(2)

# Replace NaN by the column's mean
data_bruno.fillna(data_bruno.mean(), inplace = True)

# Confirm that there are no more 'NaN'
data_bruno.isna().sum()

# Remove the 'ID' column, since it is simply an index and it does not have statistical significance.
data_bruno.drop('ID', axis = 1, inplace = True)

data_bruno.head()

# Get the proportion of the two classess in the target vector
proportion = data_bruno['class'].value_counts()/len(data_bruno['class'])


# Check for imbalanced target variables
plt.figure(figsize = (10,6))
sns.barplot(x = [2,4], y = proportion)
plt.xticks(np.arange(2),('Class 2', 'Class 4'))
plt.ylabel('Proportion')
plt.show()


sns.set_style("whitegrid")
figure = plt.figure(figsize=[10,7])
ax = sns.boxplot(data=data_bruno[['thickness', 'size', 'shape', 'Marg', 'Epith', 'bare', 'b1',
       'nucleoli', 'Mitoses']], palette="Set2", orient = 'h')
plt.tight_layout()

data_bruno.iloc[:, 1]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(projection='3d')
ax.scatter(data_bruno.iloc[:, 1], data_bruno.iloc[:, 2], data_bruno.iloc[:, 3], c=data_bruno['class'], edgecolors='k', cmap='viridis')
ax.view_init(60,70)
plt.show()

sns.pairplot(data_bruno, hue = 'class')
plt.show()


features = data_bruno[['thickness', 'size', 'shape',
                       'Marg', 'Epith', 'bare', 
                       'b1', 'nucleoli', 'Mitoses']].values
# scaled_features = StandardScaler().fit_transform(features)

reducer = umap.UMAP()

embedding = reducer.fit_transform(features)
embedding.shape

fig = plt.figure(figsize =[12,8])
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in data_bruno['class'].map({2:0, 4:1})])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Cancer dataset', fontsize=24)
plt.show()

sns.heatmap(data_bruno.drop('class', axis = 1).corr(method = 'pearson'), annot = True, cmap = 'viridis')
plt.show()

# Separate features from target
X = data_bruno.drop('class', axis = 1)
y = data_bruno['class']


# Use train_test_split to generate Training and Test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=98)

# Instantiate the LInear Support Vector Classifier
clf_linear_bruno = SVC(kernel="linear", C=0.1)

# Fit the model onto the data
clf_linear_bruno.fit(X_train, y_train)

# Get the accuracy score in the training dataset
print(f'Trainning set accuracy: {np.round(clf_linear_bruno.score(X_train, y_train), 6) * 100}%')

# Accuracy in the test dataset
print(f'Test set accuracy: {np.round(clf_linear_bruno.score(X_test, y_test), 6) * 100}%')

# `The linear model achieves an even higher accuracy in the test set.`

# Predictions using the test set
y_pred = clf_linear_bruno.predict(X_test)

# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# A good visual of the confusion matrix
plot_confusion_matrix(cm, 'Linear', [0,2], 'PuBu')

print('\t\tClassification Report\n\n',classification_report(y_test, y_pred))

datasets = [X_train, y_train, X_test, y_test]
kernels = ['rbf', 'poly', 'sigmoid']
for kernel in kernels:
    test_classifier(kernel, datasets)

# Reloading the original dataset
data_bruno_df2 = pd.read_csv('breast_cancer.csv')

# Get the head of the reloaded dataset 
data_bruno_df2.head()

data_bruno_df2['bare'].replace(to_replace = '?', value = np.nan, inplace = True)

data_bruno_df2['bare'] = data_bruno_df2['bare'].astype('float')

data_bruno_df2['bare'].dtype

data_bruno_df2.drop('ID', axis = 1, inplace = True)

X = data_bruno_df2.drop('class', axis = 1)
y = data_bruno_df2['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=98)


# Create a pipeline with transformers
num_pipeline_bruno = Pipeline([
('imputer', SimpleImputer(strategy="median")),
('std_scaler', StandardScaler()),
])

# A pipeline that calls upon the first pipeline and then the classifier
pipe_svm_bruno = Pipeline([
    ('numeric', num_pipeline_bruno),
    ('svm',  SVC(random_state = 98))
])


num_pipeline_bruno


# Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
param_grid = {'svm__kernel': ['linear', 'rbf', 'poly'],
              'svm__C': [0.01, 0.1, 1, 10, 100],
              'svm__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
              'svm__degree': [2, 3]}

param_grid


# Create a GridSearchCV object
grid_search_bruno = GridSearchCV(estimator = pipe_svm_bruno,
                                 param_grid = param_grid,
                                 scoring = 'accuracy',
                                 refit = True,
                                 n_jobs = -1,
                                 verbose = 3)


# Inspect the object
grid_search_bruno

# Get the start time
start = time.perf_counter()

# Run fit with all sets of parameters
grid_search_bruno.fit(X_train, y_train)

# The the final time of processing
end = time.perf_counter()


# Total time to run GridSearchCV
print(f'GridSearchCV processing time: {round((end-start), 2)} s')


# Best hyperparameters
print("tuned hpyerparameters :(best parameters) ", grid_search_bruno.best_params_)
print("Best Estimator :", grid_search_bruno.best_estimator_)


# Store the best model into a variable
best_model_bruno = grid_search_bruno.best_estimator_


# Inspect the object
best_model_bruno

# Make predictions with the best model
final_pred = best_model_bruno.predict(X_test)

# Accuracy in the test dataset
best_model_bruno.score(X_test, y_test)

# Print the classification Report
print('\t\tClassification Report\n\n',classification_report(y_test, final_pred))

# Import joblib to save the model
import joblib

joblib.dump(best_model_bruno, "SVC_model.pkl")

joblib.dump(pipe_svm_bruno, "full_pipeline.pkl")

# Use dill to create a copy of the whole notebook and its state
import dill

dill.dump_session('notebook_env.db')
