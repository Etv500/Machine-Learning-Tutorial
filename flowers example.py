#esercizio: http://machinelearningmastery.com/machine-learning-in-python-step-by-step/


import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


url = 'C:\Users\elvis\Desktop\Trading and Algos\python\machine _learning_example\data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
#print(dataset)

#shape of dataset
#print(dataset.shape)

#.head fa vedere i primi tot elemtni molto utile
print(dataset.head(20))

#.describe mean, std, min, max, percentili dei dati
#print(dataset.describe())

#.groupby col .size mi conta e raggruppa dati in categorie
#print(dataset.groupby('class').size())


#grafico candele
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

#grafico histogram
#dataset.hist()
#plt.show()

#scatter
#scatter_matrix(dataset)
#plt.show()

#now I will do:
#1Separate out a validation dataset.
#2Set-up the test harness to use 10-fold cross validation.
#3Build 5 different models to predict species from flower measurements
#4Select the best model.


array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20   #80% dei dati vengono usati per forecastare il restante 20% o vicerversa bo
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#You now have training data in the X_train and Y_train for preparing models and a X_validation and Y_validation sets that we can use later


print(X) #numeri
print(Y) #tipi di fiori string
#Test options and evaluation metric
seed = 7
scoring = 'accuracy'



# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
#come vedi il SVM è il modello che da il risultato migliore (0.99 accuracy)   


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()




# Make predictions on validation dataset using Knn
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

