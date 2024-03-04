# %%
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# %%
#Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# %%
# Shape of DataSet
print(dataset.shape)

# %%
# Head of DataSet
print(dataset.head(20))

# %%
# Descriptions of DataSet
#Basic statistical data only for quantitative data
print(dataset.describe())

# %%
# class distribution
print(dataset.groupby('class').size())

# %%
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, 
sharey=False)
pyplot.show()

# %%
#histogram
dataset.hist()
pyplot.show()

# %%
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# %%
# Splitting the dataset into training and validation sets
# Converting the dataset to numpy array
array = dataset.values
# Selecting independent variables (features) and dependent variable (target)
X = array[:,0:4] # Features
y = array[:,4] # Target
# Splitting the dataset into 80% training and 20% validation sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, 
test_size=0.20, random_state=1)

# %%
# Spot Check Algorithms
# Creating a list to store different machine learning models for evaluation

models = []

# Adding Logistic Regression model to the list
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

# Adding Linear Discriminant Analysis model to the list
models.append(('LDA', LinearDiscriminantAnalysis()))

# Adding K-Nearest Neighbors model to the list
models.append(('KNN', KNeighborsClassifier()))

# Adding Classification and Regression Trees model to the list
models.append(('CART', DecisionTreeClassifier()))

# Adding Naive Bayes model to the list
models.append(('NB', GaussianNB()))

# Adding Support Vector Machine model to the list
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each model in turn
# Creating empty lists to store the results and names of the models
results = []
names = []

# Looping through each model in the list
for name, model in models:
    # Using Stratified K-Folds cross-validator for splitting the data into 10 folds
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    
    # Applying cross-validation on the current model using the training data
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    
    # Storing the cross-validation results and model names
    results.append(cv_results)
    names.append(name)
    
    # Printing the mean and standard deviation of the cross-validation results for the current model
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# %%
# Compare Algorithms
# Creating a boxplot to compare the performance of different algorithms
pyplot.boxplot(results, labels=names)

# Adding a title to the plot
pyplot.title('Algorithm Comparison')

# Displaying the plot
pyplot.show()


# %%
# Make predictions on validation dataset
# Create a Support Vector Machine classifier model with automatic gamma value
model = SVC(gamma='auto')

# Train the model using the training data
model.fit(X_train, Y_train)

# Make predictions using the trained model on the validation dataset
predictions = model.predict(X_validation)


# %%
# Evaluate predictions
# Print the accuracy score of the predictions
print(accuracy_score(Y_validation, predictions))

# Print the confusion matrix of the predictions
print(confusion_matrix(Y_validation, predictions))

# Print the classification report of the predictions
print(classification_report(Y_validation, predictions))


# %%



