from __future__ import print_function
import numpy
import os
import pandas

#########################
### GENERAL FUNCTIONS ###
#########################

split = 116 # range of features considered
size = 15 # number of features considered
seed = 0
shift = 700

def readData(fulldataset=False):

    # check if running on windows or linux, because of the way directories are parsed
    dirr = os.path.dirname(os.path.realpath('__file__'))
    if os.name == 'nt':
        dirr = ''

    if not fulldataset:
        print('\nReading 10,000 samples...',end='')
        dataset = pandas.read_csv(os.path.join(dirr, "../raw data/subset/train(1-10000).csv"))
        dataset_submission = pandas.read_csv(os.path.join(dirr, "../raw data/subset/test(1-10000).csv"))
    else:
        print('\nReading full dataset...',end='')
        dataset = pandas.read_csv(os.path.join(dirr, "../raw data/full/train.csv"))
        dataset_submission = pandas.read_csv(os.path.join(dirr, "../raw data/full/test.csv"))

    print('\tFinished.')
    print('Read ' + str(dataset.shape[0]) + ' training data samples with ' + str(dataset.shape[1]) + ' attributes.')
    print('Read ' + str(dataset_submission.shape[0]) + ' testing data samples with ' + str(dataset_submission.shape[1]) + ' attributes.')

    # Save the id's for submission file, then drop unnecessary columns
    ID = dataset_submission['id']
    dataset_submission.drop('id',axis=1,inplace=True)
    dataset = dataset.iloc[:,1:]
    # Drop the first column 'id' since it just has serial numbers. Not useful in the prediction process.

    return dataset, dataset_submission, ID

def oneHotEncoding(dataset, dataset_submission, oneHotEncode=False):

    cols = dataset.columns
    labels = []

    # get unique column labels from both sets
    for i in range(0,split):
        train = dataset[cols[i]].unique()
        test = dataset_submission[cols[i]].unique()
        labels.append(list(set(train) | set(test)))

    # Import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder

    # One hot encode all categorical attributes for dataset
    cats = []
    cats_submission = []
    for i in range(0, split):
        # Label encode
        label_encoder = LabelEncoder()
        label_encoder.fit(labels[i])
        feature = label_encoder.transform(dataset.iloc[:,i])
        feature = feature.reshape(dataset.shape[0], 1)
        feature_submission = label_encoder.transform(dataset_submission.iloc[:,i])
        feature_submission = feature_submission.reshape(dataset_submission.shape[0], 1)

        # One hot encode
        if oneHotEncode:
            onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
            feature = onehot_encoder.fit_transform(feature)
            feature_submission = onehot_encoder.fit_transform(feature_submission)
        cats.append(feature)
        cats_submission.append(feature_submission)

    # Make a 2D array from a list of 1D arrays
    encoded_cats = numpy.column_stack(cats)
    encoded_cats_submission = numpy.column_stack(cats_submission)

    # Concatenate encoded attributes with continuous attributes
    dataset_encoded = numpy.concatenate((encoded_cats,dataset.iloc[:,split:].values),axis=1)
    dataset_submission_encoded = numpy.concatenate((encoded_cats_submission,dataset_submission.iloc[:,split:].values),axis=1)
    del cats
    del feature
    del dataset
    del encoded_cats
    del cats_submission
    del feature_submission
    del dataset_submission
    del encoded_cats_submission

    return dataset_encoded, dataset_submission_encoded

def transformData(data):
    return numpy.log(data+shift)

def inverseTransformData(data):
    return (numpy.exp(data)-shift)

def getResult(actual, predicted):
    actual = inverseTransformData(actual)
    predicted = inverseTransformData(predicted)
    return getResultSimple(actual, predicted)

# assumes actual and predicted values already converted
def getResultSimple(actual, predicted):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(actual,predicted) # change this to change loss fn

# list of input features to consider. for now, consider all
def getInputFeatureNames(X_train):
    i_cols_list = []
    for i in range(0,X_train.shape[1]):
        i_cols_list.append(i)
    return i_cols_list

def writeToCSV(data, filename):
    with open(filename, "wb") as csvfile:
        for row in data:
            csvfile.write(str(row) + "\n")

##############
### MODELS ###
##############

### Linear Regression (Linear algo)
# Author's best: LR, MAE=1278

def LinearReg(X_train, Y_train, X_val, Y_val, n_list, returntype='none'):

    print('\n\nLinear Regression (Linear algo)')
    print('Approx. LR, MAE=1278')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from sklearn.linear_model import LinearRegression

    model = LinearRegression(n_jobs=-1)
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
    print('MAE\t',result)

    Y_val_preds = transformData(predictedOutputx)
    Y_train_preds = model.predict(X_train[:,i_cols_list])

    if returntype == 'train':
        return Y_train_preds, -1, -1, result
    elif returntype == 'val':
        return -1, Y_val_preds, -1, result
    elif returntype == 'both':
        return Y_train_preds, Y_val_preds, -1, result
    else:
        return -1, -1, -1, result

### Ridge Regression (Linear algo)
# Author's best: alpha=1, MAE=1267

def RidgeReg(X_train, Y_train, X_val, Y_val, a_list, returntype='none'):

    print('\n\nRidge Regression (Linear algo)')
    print('Approx. alpha=1, MAE=1267')
    print('alpha\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from sklearn.linear_model import Ridge

    for alpha in a_list:
        print(str(alpha),end='\t')
        model = Ridge(alpha=alpha,random_state=seed)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = a_list[numpy.argmin(mae)]
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = Ridge(alpha=minParameter,random_state=seed)
    model.fit(X_train[:,i_cols_list],Y_train)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE

### LASSO Linear Regression (Linear algo)
# Author's best: alpha=0.001, MAE=1262.5

def LASSOLinearReg(X_train, Y_train, X_val, Y_val, a_list, returntype='none'):

    print('\n\nLASSO Linear Regression (Linear algo)')
    print('Approx. alpha=0.001, MAE=1262.5')
    print('alpha\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from sklearn.linear_model import Lasso

    for alpha in a_list:
        print(str(alpha),end='\t')
        model = Lasso(alpha=alpha,random_state=seed)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = a_list[numpy.argmin(mae)]
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = Lasso(alpha=minParameter,random_state=seed)
    model.fit(X_train[:,i_cols_list],Y_train)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE

### Elastic Net Regression (Linear algo)
# Author's best: alpha=0.001, MAE=1260

def ElasticNetReg(X_train, Y_train, X_val, Y_val, a_list, returntype='none'):
    print('\n\nElastic Net Regression (Linear algo)')
    print('Approx. alpha=0.001, MAE=1260')
    print('alpha\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from sklearn.linear_model import ElasticNet

    for alpha in a_list:
        print(str(alpha),end='\t')
        model = ElasticNet(alpha=alpha,random_state=seed)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = a_list[numpy.argmin(mae)]
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = ElasticNet(alpha=minParameter,random_state=seed)
    model.fit(X_train[:,i_cols_list],Y_train)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE

### KNN (non-linear algo)
# Author's best: n=1, MAE=1745

def KNN(X_train, Y_train, X_val, Y_val, n_list, returntype='none'):
    print('\n\nKNN (non-linear algo)')
    print('Approx. n=1, MAE=1745')
    print('n_neighbors\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from sklearn.neighbors import KNeighborsRegressor

    for n_neighbors in n_list:
        print(str(n_neighbors),end='\t')
        model = KNeighborsRegressor(n_neighbors=n_neighbors,n_jobs=-1)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = n_list[numpy.argmin(mae)]
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = KNeighborsRegressor(n_neighbors=minParameter,n_jobs=-1)
    model.fit(X_train[:,i_cols_list],Y_train)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE

### CART (non-linear algo)
# Author's best: depth=5, MAE=1741

def CART(X_train, Y_train, X_val, Y_val, d_list, returntype='none'):
    print('\n\nCART (non-linear algo)')
    print('Approx. depth=5, MAE=1741')
    print('max depth\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from sklearn.tree import DecisionTreeRegressor

    for max_depth in d_list:
        print(str(max_depth),end='\t')
        model = DecisionTreeRegressor(max_depth=max_depth,random_state=seed)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = d_list[numpy.argmin(mae)]
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = DecisionTreeRegressor(max_depth=minParameter,random_state=seed)
    model.fit(X_train[:,i_cols_list],Y_train)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE

### SVM (Non-linear algo)
# Author's best: unknown

def SVM(X_train, Y_train, X_val, Y_val, c_list, returntype='none'):
    print('\n\nSVM (Non-linear algo)')
    print('Approx. Unknown')
    print('C\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from sklearn.svm import SVR

    for C in c_list:
        print(str(C),end='\t')
        model = SVR(C=C)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = c_list[numpy.argmin(mae)]
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = SVR(C=minParameter)
    model.fit(X_train[:,i_cols_list],Y_train)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE

### Bagged Decision Trees (Bagging)
# Author's best: unknown

def BaggedDT(X_train, Y_train, X_val, Y_val, n_list, returntype='none'):
    print('\n\nBagged Decision Trees (Bagging)')
    print('Approx. Unknown')
    print('n_estimators\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor

    for n_estimators in n_list:
        print(str(n_estimators),end='\t')
        model = BaggingRegressor(n_jobs=-1,n_estimators=n_estimators)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = n_list[numpy.argmin(mae)]
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = BaggingRegressor(n_jobs=-1,n_estimators=minParameter)
    model.fit(X_train[:,i_cols_list],Y_train)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE

### Random Forest (Bagging)
# Author's best: n_est=50, MAE=1213

def RandomForest(X_train, Y_train, X_val, Y_val, n_list, returntype='none'):
    print('\n\nRandom Forest (Bagging)')
    print('Approx. n_est=50, MAE=1213')
    print('n_estimators\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from sklearn.ensemble import RandomForestRegressor

    for n_estimators in n_list:
        print(str(n_estimators),end='\t')
        model = RandomForestRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = n_list[numpy.argmin(mae)]
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = RandomForestRegressor(n_jobs=-1,n_estimators=minParameter,random_state=seed)
    model.fit(X_train[:,i_cols_list],Y_train)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE

### Extra Trees (Bagging)
# Author's best: n_est=100, MAE=1254

def ExtraTrees(X_train, Y_train, X_val, Y_val, n_list, returntype='none'):
    print('\n\nExtra Trees (Bagging)')
    print('Approx. n_est=100, MAE=1254')
    print('n_estimators\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from sklearn.ensemble import ExtraTreesRegressor

    for n_estimators in n_list:
        print(str(n_estimators),end='\t')
        model = ExtraTreesRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = n_list[numpy.argmin(mae)]
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = ExtraTreesRegressor(n_jobs=-1,n_estimators=minParameter,random_state=seed)
    model.fit(X_train[:,i_cols_list],Y_train)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE

### AdaBoost (Boosting)
# Author's best: n_est=100, MAE=1678

def AdaBoost(X_train, Y_train, X_val, Y_val, n_list, returntype='none'):
    print('\n\nAdaBoost (Boosting)')
    print('Approx. n_est=100, MAE=1678')
    print('n_estimators\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from sklearn.ensemble import AdaBoostRegressor

    for n_estimators in n_list:
        print(str(n_estimators),end='\t')
        model = AdaBoostRegressor(n_estimators=n_estimators,random_state=seed)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = n_list[numpy.argmin(mae)]
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = AdaBoostRegressor(n_estimators=minParameter,random_state=seed)
    model.fit(X_train[:,i_cols_list],Y_train)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE

### Stochastic Gradient Boosting (Boosting)
# Author's best: n_list=50, MAE=1278

def SGBoost(X_train, Y_train, X_val, Y_val, n_list, returntype='none'):
    print('\n\nStochastic Gradient Boosting (Boosting)')
    print('Approx. n_list=50, MAE=1278')
    print('n_estimators\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from sklearn.ensemble import GradientBoostingRegressor

    for n_estimators in n_list:
        print(str(n_estimators),end='\t')
        model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=seed)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = n_list[numpy.argmin(mae)]
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = GradientBoostingRegressor(n_estimators=minParameter,random_state=seed)
    model.fit(X_train[:,i_cols_list],Y_train)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE

### MLP (Deep Learning)
# Author's best: MLP, MAE=1168

def MLP(X_train, Y_train, X_val, Y_val, ParaMLP, returntype='none'):
    print('\n\nMLP (Deep Learning)')
    print('Approx. MLP, MAE=1168')
    print('Model\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.models import Sequential
    from keras.layers import Dense
    c = X_train.shape[1]+1

    # define baseline model
    def baseline(v):
        # create model
        model = Sequential()
        model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
        model.add(Dense(1, init='normal'))
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    # define smaller model
    def smaller(v):
        # create model
        model = Sequential()
        model.add(Dense(v*(c-1)/2, input_dim=v*(c-1), init='normal', activation='relu'))
        model.add(Dense(1, init='normal', activation='relu'))
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    # define deeper model
    def deeper(v):
        # create model
        model = Sequential()
        model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
        model.add(Dense(v*(c-1)/2, init='normal', activation='relu'))
        model.add(Dense(1, init='normal', activation='relu'))
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    # Optimize using dropout and decay
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
        # Compile model
        sgd = SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
        model.compile(loss='mean_absolute_error', optimizer=sgd)
        return model

    # define decay model
    def decay(v):
        # create model
        model = Sequential()
        model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
        model.add(Dense(1, init='normal', activation='relu'))
        # Compile model
        sgd = SGD(lr=0.1,momentum=0.8,decay=0.01,nesterov=False)
        model.compile(loss='mean_absolute_error', optimizer=sgd)
        return model

    est_list = [('MLP',baseline),('smaller',smaller),('deeper',deeper),('dropout',dropout),('decay',decay)]

    for name, est in est_list:
        print(str(name),end='\t')
        model = KerasRegressor(build_fn=est, v=1, nb_epoch=10, verbose=0)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = est_list[numpy.argmin(mae)][1]
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = KerasRegressor(build_fn=minParameter, v=1, nb_epoch=10, verbose=0)
    model.fit(X_train[:,i_cols_list],Y_train)

    # since minParameter is usually int and not string
    minParameter = numpy.argmin(mae)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE

### XGBoost

def XGBoost(X_train, Y_train, X_val, Y_val, n_list, returntype='none'):
    print('\n\nXGBoost')
    print('Approx. n=1000, MAE=1169')
    print('n_estimators\tMAE')

    i_cols_list = getInputFeatureNames(X_train)
    mae = []

    from xgboost import XGBRegressor

    for n_estimators in n_list:
        print(str(n_estimators),end='\t')
        model = XGBRegressor(
            n_estimators=n_estimators,
            num_boost_round=200,
            gamma=0.2,
            max_depth=8,
            min_child_weight=6,
            colsample_bytree=0.6,
            subsample=0.9,
            eta=0.07
            ) # see https://github.com/dnkirill/allstate_capstone/blob/master/part2_xgboost.ipynb
        model.fit(X_train[:,i_cols_list],Y_train)
        result = getResult(Y_val, model.predict(X_val[:,i_cols_list]))
        print(result)
        mae = numpy.append(mae, result)

    # find parameter with minimum MAE
    # train model with that parameter
    minMAE = mae[numpy.argmin(mae)]
    minParameter = n_list[numpy.argmin(mae)]
##    X = numpy.concatenate((X_train,X_val),axis=0)
##    Y = numpy.concatenate((Y_train,Y_val),axis=0)
##    X_train = X
##    Y_train = Y
    print('Min MAE = ' + str(numpy.around(minMAE,3)) + ' at ' + str(minParameter))
    model = XGBRegressor(
        n_estimators=minParameter,
        num_boost_round=200,
        gamma=0.2,
        max_depth=8,
        min_child_weight=6,
        colsample_bytree=0.6,
        subsample=0.9,
        eta=0.07
        ) # see https://github.com/dnkirill/allstate_capstone/blob/master/part2_xgboost.ipynb
    model.fit(X_train[:,i_cols_list],Y_train)

    if returntype == 'train':
        return model.predict(X_train[:,i_cols_list]), -1, minParameter, minMAE
    elif returntype == 'val':
        return -1, model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    elif returntype == 'both':
        return model.predict(X_train[:,i_cols_list]), model.predict(X_val[:,i_cols_list]), minParameter, minMAE
    else:
        return -1, -1, minParameter, minMAE
