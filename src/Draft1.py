# The purpose of this draft is to assess the score of each possible model.

from __future__ import print_function
print('Importing libraries...')
import keras
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn as sns
from xgboost import XGBRegressor
import os
import sys
import socket

print(sys.version)
print('Finished importing libraries.')

##########################
### USER CONFIGURATION ###
##########################

# choose full dataset or 10,000 samples
fulldataset = False

#  booleans to quickly enable / disable code
datadescription = False # Disable to disable printing and plotting of data statistics
plotResults = False # Disable if you want to display MAE only, and don't want to plot results
makePredictions = False

EvalLinearReg = False
EvalRidgeReg = False
EvalLASSOLinearReg = False
EvalElasticNetReg = False
EvalKNN = False
EvalCART = False
EvalSVM = False
EvalBaggedDecisionTrees = False
EvalRandomForest = False
EvalExtraTrees = False
EvalAdaBoost = False
EvalSGBoost = False
EvalMLP = False
EvalXGBoost = False

# alt4 to enable, alt3 to disable quickly
##EvalLinearReg = True
##EvalRidgeReg = True
##EvalLASSOLinearReg = True
##EvalElasticNetReg = True
##EvalKNN = True
##EvalCART = True
##EvalSVM = True
##EvalBaggedDecisionTrees = True
##EvalRandomForest = True
##EvalExtraTrees = True
##EvalAdaBoost = True
##EvalSGBoost = True
##EvalMLP = True
##EvalXGBoost = True

# Input parameters
ParaRidgeReg = numpy.array([1,5,10,30,50,60]);
ParaLASSOLinearReg = numpy.array([0.00003,0.00004,0.00005]);
ParaElasticNetReg = numpy.array([0.002,0.003,0.004,0.005]);
ParaKNN = numpy.array([1,2,5]);
ParaCART = numpy.array([1,3,5,7,9]);
ParaSVM = numpy.array([0.1,0.5,1,3,5,10,50]);
ParaBaggedDecisionTrees = numpy.array([1,2,3,4,5]);
ParaRandomForest = numpy.array([30,50,70,90,110,130]);
ParaExtraTrees = numpy.array([70,90,110,130,150,170,190]);
ParaAdaBoost = numpy.array([80,100,120,160,180]);
ParaSGBoost = numpy.array([350,400,500]);
ParaXGBoost = numpy.array([800]);

##################
### END CONFIG ###
##################

# Read datasets

# check if running on windows or linux, because of the way directories are parsed
dirr = os.path.dirname(os.path.realpath('__file__'))
if os.name == 'nt':
    dirr = ''

if not fulldataset:
    print('Reading 10,000 samples...')
    dataset = pandas.read_csv(os.path.join(dirr, "../raw data/subset/train(1-10000).csv"))
    dataset_test = pandas.read_csv(os.path.join(dirr, "../raw data/subset/test(1-10000).csv"))
else:
    print('Reading full dataset...')
    dataset = pandas.read_csv(os.path.join(dirr, "../raw data/full/train.csv"))
    dataset_test = pandas.read_csv(os.path.join(dirr, "../raw data/full/test.csv"))

print('Finished reading datasets.')
print('Read ' + str(dataset.shape[0]) + ' training data samples with ' + str(dataset.shape[1]) + ' attributes.')
print('Read ' + str(dataset_test.shape[0]) + ' testing data samples with ' + str(dataset_test.shape[1]) + ' attributes.')

# Save the id's for submission file, then drop unnecessary columns
ID = dataset_test['id']
dataset_test.drop('id',axis=1,inplace=True)

# Print all rows and columns. Dont hide any
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

# Display the first five rows to get a feel of the data
if datadescription:
    print(dataset.head(5))

# Learning: cat1 to cat116 contain alphabets

###############################
### DATA STATISTICS - SHAPE ###
###############################

#  Size of the dataframe

if datadescription:
    print(dataset.shape)

#  We can see that there are 188318 instances having 132 attributes

# Drop the first column 'id' since it just has serial numbers. Not useful in the prediction process.
dataset = dataset.iloc[:,1:]

# Learning: Data is loaded successfully as dimensions match the data description

#####################################
### DATA STATISTICS - DESCRIPTION ###
#####################################

#  Statistical description

if datadescription:
    print(dataset.describe())

#  Learning:
#  No attribute in continuous columns is missing as count is 188318 for all, all rows can be used
#  No negative values are present. Tests such as chi2 can be used
#  Statistics not displayed for categorical data

##############################
### DATA STATISTICS - SKEW ###
##############################

#  Skewness of the distribution

if datadescription:
    print(dataset.skew())

#  Values close to 0 show less skew
#  loss shows the highest skew. Let us visualize it

################################################
### DATA VISUALIZATION - BOX & DENSITY PLOTS ###
################################################

#  We will visualize all the continuous attributes using Violin Plot - a combination of box and density plots

# range of features considered
split = 116

# number of features considered
size = 15

# create a dataframe with only continuous features
data=dataset.iloc[:,split:]

# get the names of all the columns
cols=data.columns

# Plot violin for all attributes in a 7x2 grid
n_cols = 2
n_rows = 7

if datadescription:
    for i in range(n_rows):
        fg,ax = plt.subplots(nrows=1,ncols=n_cols,figsize=(12, 8))
        for j in range(n_cols):
            sns.violinplot(y=cols[i*n_cols+j], data=dataset, ax=ax[j])

# cont1 has many values close to 0.5
# cont2 has a pattern where there a several spikes at specific points
# cont5 has many values near 0.3
# cont14 has a distinct pattern. 0.22 and 0.82 have a lot of concentration
# loss distribution must be converted to normal

#############################################
### DATA TRANSFORMATION - SKEW CORRECTION ###
#############################################

# define functions for transforming / inverse transforming the loss
# so that we don't have to manually replace functions when we decide to change
# the transformation function
# so throughout this program, transformData() and inverseTransformData()
# will be used instead of whatever function you're using to transform

shift=700
def transformData(data):
    return numpy.log(data+shift)
def inverseTransformData(data):
    return (numpy.exp(data)-shift)

# transformed loss = ln(originalloss^0.25)
# original loss = (e^(transformedloss))^4
##power=0.25
##def transformData(data):
##    return numpy.log(numpy.power(data,0.25))
##def inverseTransformData(data):
##    return (numpy.power(numpy.exp(data),4))

# transformed loss = ln(shift+original loss)
# original loss = e^(transformed loss)-shift
# author's shift is 1
dataset["loss"] = transformData(dataset["loss"])

if datadescription:
    print(dataset.describe())
    print(dataset.skew())
    # visualize the transformed column
    sns.violinplot(data=dataset,y="loss")
    plt.show()
    plt.close('all')

# Plot shows that skew is corrected to a large extent

#####################################
### DATA INTERATION - CORRELATION ###
#####################################

#  Correlation tells relation between two attributes.
#  Correlation requires continous data. Hence, ignore categorical data

#  Calculates pearson co-efficient for all combinations
data_corr = data.corr()

#  Set the threshold to select only highly correlated attributes
threshold = 0.5

#  List of pairs along with correlation above threshold
corr_list = []

# Search for the highly correlated pairs
for i in range(0,size): # for 'size' features
    for j in range(i+1,size): # avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) # store correlation and columns index

# Sort to show higher ones first
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

if datadescription:
    # Print correlations and column names
    for v,i,j in s_corr_list:
        print ("%s and %s = %.2f" % (cols[i],cols[j],v))

#  Strong correlation is observed between the following pairs
#  This represents an opportunity to reduce the feature set through transformations such as PCA

######################################
### DATA INTERATION - SCATTER PLOT ###
######################################

if datadescription:
    #  Scatter plot of only the highly correlated pairs
    for v,i,j in s_corr_list:
        sns.pairplot(dataset, size=6, x_vars=cols[i],y_vars=cols[j] )
        plt.show()
    plt.close('all')

# cont11 and cont12 give an almost linear pattern...one must be removed
# cont1 and cont9 are highly correlated ...either of them could be safely removed
# cont6 and cont10 show very good correlation too

###################################################
### DATA VISUALIZATION - CATEGORICAL ATTRIBUTES ###
###################################################

#  Count of each label in each category

# names of all the columns
cols = dataset.columns

# Plot count plot for all attributes in a 29x4 grid
n_cols = 4
n_rows = 29
if datadescription:
    for i in range(n_rows):
        fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(12, 8))
        for j in range(n_cols):
            sns.countplot(x=cols[i*n_cols+j], data=dataset, ax=ax[j])

# cat1 to cat72 have only two labels A and B. In most of the cases, B has very few entries
# cat73 to cat 108 have more than two labels
# cat109 to cat116 have many labels

#  This code MUST be run before running any models

########################
### DATA PREPARATION ###
########################

############################################
### One-Hot encoding of categorical data ###
############################################

# cat1 to cat116 have strings. The ML algorithms we are going to study require numberical data
# One-hot encoding converts an attribute to a binary vector

# Variable to hold the list of variables for an attribute in the train and test data
labels = []

for i in range(0,split):
    train = dataset[cols[i]].unique()
    test = dataset_test[cols[i]].unique()
    labels.append(list(set(train) | set(test)))

del dataset_test

#  Import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#  One hot encode all categorical attributes
cats = []
for i in range(0, split):
    # Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset.iloc[:,i])
    feature = feature.reshape(dataset.shape[0], 1)
    # One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

#  Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

if datadescription:
    #  Print the shape of the encoded data
    print(encoded_cats.shape)

# Concatenate encoded attributes with continuous attributes
dataset_encoded = numpy.concatenate((encoded_cats,dataset.iloc[:,split:].values),axis=1)
del cats
del feature
del dataset
del encoded_cats
if datadescription:
    print(dataset_encoded.shape)

#######################################
### Split into train and validation ###
#######################################

# get the number of rows and columns
r, c = dataset_encoded.shape

# create an array which has indexes of columns
i_cols = []
for i in range(0,c-1):
    i_cols.append(i)

# Y is the target column, X has the rest
X = dataset_encoded[:,0:(c-1)]
Y = dataset_encoded[:,(c-1)]
del dataset_encoded

# Validation chunk size
val_size = 0.1

# Use a common seed in all experiments so that same chunk is used for validation
seed = 0

# Split the data into chunks
from sklearn import cross_validation
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)
del X
del Y

# final datasets:   X_train, X_val = input features for training and validation
#                   Y_train, Y_val = output values for training and validation

# All features
X_all = []

# List of combinations
comb = []

# Dictionary to store the MAE for all algorithms
mae = []

# Add this version of X to the list
n = "All"
X_all.append([n, i_cols])

##################
### EVALUATION ###
##################

# calculating error based on calculated output depends on:
# 1. restoring calculated values to original scale, which depends on how you first transformed it.
# eg author used loss = log(loss+1) = numpy.log1p(loss) so inverse is loss = e^(loss)-1 = numpy.expm1(loss)
# 2. evaluation metric used. here it's MAE
from sklearn.metrics import mean_absolute_error

# assumes actual and predicted values already converted
def getResultSimple(actual, predicted):
    return mean_absolute_error(actual,predicted) # change this to change loss fn

# assumes actual and predicted values already converted
def getResult(actual, predicted):
    actual = inverseTransformData(actual)
    predicted = inverseTransformData(predicted)
    return getResultSimple(actual, predicted)

### Linear Regression (Linear algo)
#  Author's best: LR, MAE=1278

def LinearReg():

    print('\n\nLinear Regression (Linear algo)')
    print('Author\'s best: LR, MAE=1278')
    print('\tMAE')

    from sklearn.linear_model import LinearRegression

    # Set the base model
    model = LinearRegression(n_jobs=-1)
    algo = "LR"

    # Accuracy of the model using all features
    for name,i_cols_list in X_all:
        # print('Name = ' + str(name) + '; i_cols_list = ' + str(i_cols_list))
        model.fit(X_train[:,i_cols_list],Y_train)

        predictedOutput = inverseTransformData(model.predict(X_val[:,i_cols_list]))

        # Check output
        if numpy.isnan(predictedOutput).any():
            print('WARNING: some predicted output values are NaN, and will be replaced with 0!')
        if numpy.isinf(predictedOutput).any():
            print('WARNING: some predicted output values are inf, and will be replaced with ' + str(numpy.finfo('float64').max) + '!')

       # Sanitize predicted output so it doesn't contain inf or nan values
        predictedOutputx = []
        for i in predictedOutput:
            if numpy.isinf(i):
                # substitute inf values with largest possible representable number
                predictedOutputx = numpy.append(predictedOutputx,numpy.finfo('float64').max)
            elif numpy.isnan(i):
                # substitute nan values with 0
                predictedOutputx = numpy.append(predictedOutputx,0)
            else:
                predictedOutputx = numpy.append(predictedOutputx,i)

        result = getResultSimple(inverseTransformData(Y_val), predictedOutputx)
        mae.append(result)
        print('\t' + str(result))
    comb.append(algo)

    if plotResults:
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()
        plt.close('all')

### Ridge Regression (Linear algo)
#  Author's best: alpha=1, MAE=1267

def RidgeReg(a_list):

    print('\n\nRidge Regression (Linear algo)')
    print('Author\'s best: alpha=1, MAE=1267')
    print('alpha\tMAE')

    from sklearn.linear_model import Ridge

    for alpha in a_list:
        print(str(alpha), end='')

        # Set the base model
        model = Ridge(alpha=alpha,random_state=seed)
        algo = "Ridge"

        # Accuracy of the model using all features
        for name,i_cols_list in X_all:
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            print('\t' + str(result))

        comb.append(algo + " %s" % alpha )

    # Result obtained by running the algo for alpha=1.0
    if (len(a_list)==0):
        mae.append(1267.5)
        comb.append("Ridge" + " %s" % 1.0 )

    if plotResults:
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()
        plt.savefig('MAE of Ridge')

### LASSO Linear Regression (Linear algo)
#  Author's best: alpha=0.001, MAE=1262.5

def LASSOLinearReg(a_list):

    print('\n\nLASSO Linear Regression (Linear algo)')
    print('Author\'s best: alpha=0.001, MAE=1262.5')
    print('alpha\tMAE')

    from sklearn.linear_model import Lasso

    for alpha in a_list:
        # Set the base model
        model = Lasso(alpha=alpha,random_state=seed)
        print(str(alpha),end='')

        algo = "Lasso"

        # Accuracy of the model using all features
        for name,i_cols_list in X_all:
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            print('\t' + str(result))

        comb.append(algo + " %s" % alpha )

    # Result obtained by running the algo for alpha=0.001
    if (len(a_list)==0):
        mae.append(1262.5)
        comb.append("Lasso" + " %s" % 0.001 )
    # Set figure size
    plt.rc("figure", figsize=(25, 10))

    if plotResults:
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()

### Elastic Net Regression (Linear algo)
#  Author's best: alpha=0.001, MAE=1260

def ElasticNetReg(a_list):
    print('\n\nElastic Net Regression (Linear algo)')
    print('Author\'s best: alpha=0.001, MAE=1260')
    print('alpha\tMAE')

    from sklearn.linear_model import ElasticNet

    for alpha in a_list:
        # Set the base model
        model = ElasticNet(alpha=alpha,random_state=seed)
        print(str(alpha), end='')
        algo = "Elastic"

        # Accuracy of the model using all features
        for name,i_cols_list in X_all:
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            print('\t' + str(result))

        comb.append(algo + " %s" % alpha )

    if (len(a_list)==0):
        mae.append(1260)
        comb.append("Elastic" + " %s" % 0.001 )

    if plotResults:
        # Set figure size
        plt.rc("figure", figsize=(25, 10))
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()

### KNN (non-linear algo)
#  Author's best: n=1, MAE=1745

def KNN(n_list):
    print('\n\nKNN (non-linear algo)')
    print('Author\'s best: n=1, MAE=1745')
    print('n_neighbors\tMAE')

    from sklearn.neighbors import KNeighborsRegressor

    for n_neighbors in n_list:
        # Set the base model
        model = KNeighborsRegressor(n_neighbors=n_neighbors,n_jobs=-1)
        print(str(n_neighbors), end='')
        algo = "KNN"

        # Accuracy of the model using all features
        for name,i_cols_list in X_all:
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            print('\t' + str(result))

        comb.append(algo + " %s" % n_neighbors )

    if (len(n_list)==0):
        mae.append(1745)
        comb.append("KNN" + " %s" % 1 )

    if plotResults:
        # Set figure size
        plt.rc("figure", figsize=(25, 10))
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()

### CART (non-linear algo)
#  Author's best: depth=5, MAE=1741

def CART(d_list):
    print('\n\nCART (non-linear algo)')
    print('Author\'s best: depth=5, MAE=1741')
    print('max depth\tMAE')
    from sklearn.tree import DecisionTreeRegressor

    for max_depth in d_list:
        # Set the base model
        model = DecisionTreeRegressor(max_depth=max_depth,random_state=seed)
        print(str(max_depth), end='')
        algo = "CART"

        # Accuracy of the model using all features
        for name,i_cols_list in X_all:
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            #  print('\t' + str(result))
            print('\t' + str(result))

        comb.append(algo + " %s" % max_depth )

    if (len(d_list)==0):
        mae.append(1741)
        comb.append("CART" + " %s" % 5 )

    if plotResults:
        # Set figure size
        plt.rc("figure", figsize=(25, 10))
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()

### SVM (Non-linear algo)
#  Author's best: unknown

def SVM(c_list):
    print('\n\nSVM (Non-linear algo)')
    print('Author\'s best: Unknown')
    print('C\t\tMAE')
    from sklearn.svm import SVR

    for C in c_list:
        # Set the base model
        model = SVR(C=C)
        print(str(C), end='')
        algo = "SVM"

        # Accuracy of the model using all features
        for name,i_cols_list in X_all:
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            print('\t' + str(result))

        comb.append(algo + " %s" % C )

    # Set figure size
    plt.rc("figure", figsize=(25, 10))

    if plotResults:
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()

### Bagged Decision Trees (Bagging)
#  Author's best: unknown

def BaggedDecisionTrees(n_list):
    print('\n\nBagged Decision Trees (Bagging)')
    print('Author\'s best: Unknown')
    print('n_estimators\tMAE')
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor

    for n_estimators in n_list:
        # Set the base model
        model = BaggingRegressor(n_jobs=-1,n_estimators=n_estimators)
        print(str(n_estimators),end='')
        algo = "Bag"

        # Accuracy of the model using all features
        for name,i_cols_list in X_all:
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            print('\t' + str(result))

        comb.append(algo + " %s" % n_estimators )

    if plotResults:
        # Set figure size
        plt.rc("figure", figsize=(25, 10))
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()

### Random Forest (Bagging)
#  Author's best: n_est=50, MAE=1213

def RandomForest(n_list):
    print('\n\nRandom Forest (Bagging)')
    print('Author\'s best: n_est=50, MAE=1213')
    print('n_estimators\tMAE')

    from sklearn.ensemble import RandomForestRegressor

    for n_estimators in n_list:
        # Set the base model
        model = RandomForestRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)
        print(str(n_estimators),end='')
        algo = "RF"

        # Accuracy of the model using all features
        for name,i_cols_list in X_all:
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            print('\t' + str(result))

        comb.append(algo + " %s" % n_estimators )

    if (len(n_list)==0):
        mae.append(1213)
        comb.append("RF" + " %s" % 50 )

    if plotResults:
        # Set figure size
        plt.rc("figure", figsize=(25, 10))
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()

### Extra Trees (Bagging)
#  Author's best: n_est=100, MAE=1254

def ExtraTrees(n_list):
    print('\n\nExtra Trees (Bagging)')
    print('Author\'s best: n_est=100, MAE=1254')
    print('n_estimators\tMAE')
    from sklearn.ensemble import ExtraTreesRegressor

    for n_estimators in n_list:
        # Set the base model
        model = ExtraTreesRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)
        print(str(n_estimators),end='')
        algo = "ET"

        # Accuracy of the model using all features
        for name,i_cols_list in X_all:
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            print('\t' + str(result))

        comb.append(algo + " %s" % n_estimators )

    if (len(n_list)==0):
        mae.append(1254)
        comb.append("ET" + " %s" % 100 )

    if plotResults:
        # Set figure size
        plt.rc("figure", figsize=(25, 10))
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()

### AdaBoost (Boosting)
#  Author's best: n_est=100, MAE=1678

def AdaBoost(n_list):
    print('\n\nAdaBoost (Boosting)')
    print('Author\'s best: n_est=100, MAE=1678')
    print('n_estimators\tMAE')
    from sklearn.ensemble import AdaBoostRegressor

    for n_estimators in n_list:
        # Set the base model
        model = AdaBoostRegressor(n_estimators=n_estimators,random_state=seed)
        print(str(n_estimators),end='')
        algo = "Ada"

        # Accuracy of the model using all features
        for name,i_cols_list in X_all:
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            print('\t' + str(result))

        comb.append(algo + " %s" % n_estimators )

    if (len(n_list)==0):
        mae.append(1678)
        comb.append("Ada" + " %s" % 100 )

    if plotResults:
        # Set figure size
        plt.rc("figure", figsize=(25, 10))
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()

### Stochastic Gradient Boosting (Boosting)
#  Author's best: n_list=50, MAE=1278

def SGBoost(n_list):
    print('\n\nStochastic Gradient Boosting (Boosting)')
    print('Author\'s best: n_list=50, MAE=1278')
    print('n_estimators\tMAE')
    from sklearn.ensemble import GradientBoostingRegressor

    for n_estimators in n_list:
        # Set the base model
        model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=seed)
        print(str(n_estimators),end='')
        algo = "SGB"

        # Accuracy of the model using all features
        for name,i_cols_list in X_all:
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            print('\t' + str(result))

        comb.append(algo + " %s" % n_estimators )

    if (len(n_list)==0):
        mae.append(1278)
        comb.append("SGB" + " %s" % 50 )

    if plotResults:
        # Set figure size
        plt.rc("figure", figsize=(25, 10))
        #  Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        #  Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        #  Plot the accuracy for all combinations
        plt.show()

### MLP (Deep Learning)
#  Author's best: MLP, MAE=1168

def MLP():
    print('\n\nMLP (Deep Learning)')
    print('Author\'s best: MLP, MAE=1168')
    print('Model\tMAE')

    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.models import Sequential
    from keras.layers import Dense

    #  define baseline model
    def baseline(v):
        #  create model
        model = Sequential()
        model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
        model.add(Dense(1, init='normal'))
        #  Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    #  define smaller model
    def smaller(v):
        #  create model
        model = Sequential()
        model.add(Dense(v*(c-1)/2, input_dim=v*(c-1), init='normal', activation='relu'))
        model.add(Dense(1, init='normal', activation='relu'))
        #  Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    #  define deeper model
    def deeper(v):
        #  create model
        model = Sequential()
        model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
        model.add(Dense(v*(c-1)/2, init='normal', activation='relu'))
        model.add(Dense(1, init='normal', activation='relu'))
        #  Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    #  Optimize using dropout and decay
    from keras.optimizers import SGD
    from keras.layers import Dropout
    from keras.constraints import maxnorm

    def dropout(v):
        # create model
        model = Sequential()
        model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu',W_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(v*(c-1)/2, init='normal', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(1, init='normal', activation='relu'))
        #  Compile model
        sgd = SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
        model.compile(loss='mean_absolute_error', optimizer=sgd)
        return model

    #  define decay model
    def decay(v):
        #  create model
        model = Sequential()
        model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
        model.add(Dense(1, init='normal', activation='relu'))
        #  Compile model
        sgd = SGD(lr=0.1,momentum=0.8,decay=0.01,nesterov=False)
        model.compile(loss='mean_absolute_error', optimizer=sgd)
        return model

    est_list = []
    # uncomment the below if you want to run the algo
    est_list = [('MLP',baseline),('smaller',smaller),('deeper',deeper),('dropout',dropout),('decay',decay)]

    for name, est in est_list:
        print(str(name),end='')
        algo = name

        # Accuracy of the model using all features
        for m,i_cols_list in X_all:
            model = KerasRegressor(build_fn=est, v=1, nb_epoch=10, verbose=0)
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            print('\t' + str(result))

        comb.append(algo )

    if (len(est_list)==0):
        mae.append(1168)
        comb.append("MLP" + " baseline" )

    if plotResults:
        # Set figure size
        plt.rc("figure", figsize=(25, 10))
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()



### XGBoost
#  Author's best: n=1000, MAE=1169

def XGBoost(n_list):
    print('\n\nXGBoost')
    print('Author\'s best: n=1000, MAE=1169')
    print('n_estimators\tMAE')
    from xgboost import XGBRegressor

    for n_estimators in n_list:
        # Set the base model
        model = XGBRegressor(
            n_estimators=800,
            seed=seed
            )
        print(str(n_estimators),end='')
        algo = "XGB"

        # Accuracy of the model using all features
        for name,i_cols_list in X_all:
            model.fit(X_train[:,i_cols_list],Y_train)
            result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
            mae.append(result)
            print('\t' + str(result))

        #comb.append(algo + " %s" % n_estimators )

    if (len(n_list)==0):
        mae.append(1169)
        comb.append("XGB" + " %s" % 1000 )

    if plotResults:
        # Set figure size
        plt.rc("figure", figsize=(25, 10))
        # Plot the MAE of all combinations
        fig, ax = plt.subplots()
        plt.plot(mae)
        # Set the tick names to names of combinations
        ax.set_xticks(range(len(comb)))
        ax.set_xticklabels(comb,rotation='vertical')
        # Plot the accuracy for all combinations
        plt.show()

if EvalLinearReg:		
	LinearReg();	
if EvalRidgeReg:		
	RidgeReg(ParaRidgeReg);
if EvalLASSOLinearReg:		
	LASSOLinearReg(ParaLASSOLinearReg);
if EvalElasticNetReg:		
	ElasticNetReg(ParaElasticNetReg);
if EvalKNN:		
	KNN(ParaKNN);
if EvalCART:		
	CART(ParaCART);
if EvalSVM:		
	SVM(ParaSVM);
if EvalBaggedDecisionTrees:		
	BaggedDecisionTrees(ParaBaggedDecisionTrees);
if EvalRandomForest:		
	RandomForest(ParaRandomForest);
if EvalExtraTrees:		
	ExtraTrees(ParaExtraTrees);
if EvalAdaBoost:		
	AdaBoost(ParaAdaBoost);
if EvalSGBoost:		
	SGBoost(ParaSGBoost);
if EvalMLP:		
	MLP();	
if EvalXGBoost:		
	XGBoost(ParaXGBoost);

if makePredictions:
    print('Making predictions...')
    #  Make predictions using XGB as it gave the best estimated performance

    X = numpy.concatenate((X_train,X_val),axis=0)
    del X_train
    del X_val
    Y = numpy.concatenate((Y_train,Y_val),axis=0)
    del Y_train
    del Y_val

    n_estimators=100

    # Best model definition
    best_model = XGBRegressor(n_estimators=n_estimators,seed=seed)
    best_model.fit(X,Y)
    del X
    del Y

    # check if running on windows or linux, because of the way directories are parsed
    dirr = os.path.dirname(os.path.realpath('__file__'))
    if os.name == 'nt':
        dirr = ''
        
    if not fulldataset:
        dataset_test = pandas.read_csv(os.path.join(dirr, "../raw data/subset/test(1-10000).csv"))
    else:
        dataset_test = pandas.read_csv(os.path.join(dirr, "../raw data/full/test.csv"))

    # dataset_test = pandas.read_csv(os.path.join(dirr, "../raw data/full/test.csv"))
        
    # Drop unnecessary columns
    ID = dataset_test['id']
    dataset_test.drop('id',axis=1,inplace=True)

    # One hot encode all categorical attributes
    cats = []
    for i in range(0, split):
        # Label encode
        label_encoder = LabelEncoder()
        label_encoder.fit(labels[i])
        feature = label_encoder.transform(dataset_test.iloc[:,i])
        feature = feature.reshape(dataset_test.shape[0], 1)
        # One hot encode
        onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
        feature = onehot_encoder.fit_transform(feature)
        cats.append(feature)

    #  Make a 2D array from a list of 1D arrays
    encoded_cats = numpy.column_stack(cats)

    del cats

    # Concatenate encoded attributes with continuous attributes
    X_test = numpy.concatenate((encoded_cats,dataset_test.iloc[:,split:].values),axis=1)

    del encoded_cats
    del dataset_test

    # Make predictions using the best model
    predictions = inverseTransformData(best_model.predict(X_test))
    del X_test
    #  Write submissions to output file in the correct format
    with open("submission.csv", "w") as subfile:
        subfile.write("id,loss\n")
        for i, pred in enumerate(list(predictions)):
            subfile.write("%s,%s\n"%(ID[i],pred))

print('end')
