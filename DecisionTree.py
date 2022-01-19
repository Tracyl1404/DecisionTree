import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
#if true, data is a pandas dataframe. Will classify it
#dataset=datasets.load_iris() #preset data

#fit a cart model to the data
#model = DecisionTreeClassifier()
#model.fit(dataset.data, dataset.target)
#print(model)

#make predictions
#expected = dataset.target
#predicted = model.predict(dataset.data)

#summarize the fit of the model
#print(metrics.classification_report(expected,predicted))
#print(metrics.confusion_matrix(expected,predicted))


#Iris classification
#load through URL
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
dataset = pd.read_csv(url, names=attributes)
dataset.columns = attributes
print(dataset)
print(dataset.dtypes)
# number of instances in each class/species
print(dataset.groupby('species').size())
#take out a test set
#set training size vs testing size to 60 - 40
train, test = train_test_split(dataset, test_size = 0.4, stratify= dataset['species'], random_state=42)
#what's the number of instances in each class in training data
print(train.groupby('species').size())
#histograms per attributes for all species
n_bins = 10
fig1, axs = plt.subplots(2,2)
axs[0,0].hist(train['sepal_length'], bins = n_bins);
axs[0,0].set_title('sepal_length');
axs[0,1].hist(train['sepal_width'], bins = n_bins);
axs[0,1].set_title('sepal_width');
axs[1,0].hist(train['petal_length'], bins = n_bins);
axs[1,0].set_title('petal_length');
axs[1,1].hist(train['petal_width'], bins = n_bins);
axs[1,1].set_title('petal_width');
fig1.tight_layout(pad=0.5); #add some space
#from the hist, we can see that the mean estimate for petal length and width is smaller than the sepal length/width
#lets do a box plot for attributes vs species
fig, axs = plt.subplots(2, 2)
fn = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
cn = ['setosa', 'versicolor', 'virginica']
sns.boxplot(x = 'species', y = 'sepal_length', data=train, order = cn, ax = axs[0,0]);
sns.boxplot(x = 'species', y = 'sepal_width', data=train, order = cn, ax = axs[0,1]);
sns.boxplot(x = 'species', y = 'petal_length', data=train, order = cn, ax = axs[1,0]);
sns.boxplot(x = 'species', y = 'petal_width', data=train,  order = cn, ax = axs[1,1]);
plt.show()