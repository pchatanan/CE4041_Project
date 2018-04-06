## Ensemble learning

from __future__ import print_function
print('Importing libraries...',end='')
import keras
import warnings
warnings.filterwarnings('ignore')
import numpy
import pandas
import xgboost as xgb
from xgboost import XGBRegressor
import os
import sys
import ce4041
print('\tFinished.')
print('Python ' + str(sys.version))

##########################

fulldataset = True

makePredictions = False
ensembleLearnerXGBoost = True
ensembleLearner = False

EvalLinearReg = False
EvalRidgeReg = False
EvalLASSOLinearReg = False
EvalElasticNetReg = False
EvalKNN = False
EvalCART = False
EvalSVM = False
EvalBaggedDT = False
EvalRandomForest = False
EvalExtraTrees = False
EvalAdaBoost = False
EvalSGBoost = False
EvalMLP = False
EvalXGBoost = False

# alt4 to enable, alt3 to disable quickly
EvalLinearReg = True
EvalRidgeReg = True
EvalLASSOLinearReg = True
EvalElasticNetReg = True
EvalKNN = True
EvalCART = True
EvalSVM = True
##EvalBaggedDT = True
EvalRandomForest = True
EvalExtraTrees = True
EvalAdaBoost = True
EvalSGBoost = True
EvalMLP = True
EvalXGBoost = True

### Input parameters
ParaLinearReg = numpy.array([])
ParaRidgeReg = numpy.array([100,150,200,300]);
ParaLASSOLinearReg = numpy.array([0.001,0.002,0.003,0.005]);
ParaElasticNetReg = numpy.array([0.0001,0.0005,0.001,0.003]);
ParaKNN = numpy.array([5, 7, 9,11]);
ParaCART = numpy.array([5,7,9,11,13]);
ParaSVM = numpy.array([1,3,5,10]);
ParaBaggedDT = numpy.array([1,2,3,4,5]);
ParaRandomForest = numpy.array([70,90,110]);
ParaExtraTrees = numpy.array([130,150,170,190,210,230]);
ParaAdaBoost = numpy.array([60,80,100,120,160]);
ParaSGBoost = numpy.array([80,100,120,150,180]);
ParaMLP = numpy.array([])
ParaXGBoost = numpy.array([150,350,550,750,950,1150,1500]);

##########################

dataset, dataset_submission, dataset_submission_ID = ce4041.readData(fulldataset=fulldataset)

print('\nTransforming outputs using log(loss+' + str(ce4041.shift) + ')')
dataset["loss"] = ce4041.transformData(dataset["loss"])

print('One-hot encoding of categorical inputs')
dataset_encoded, dataset_submission_encoded = ce4041.oneHotEncoding(dataset, dataset_submission)
del dataset
del dataset_submission

print('Splitting data into inputs/outputs')
X = dataset_encoded[:,0:(dataset_encoded.shape[1]-1)]
Y = dataset_encoded[:,(dataset_encoded.shape[1]-1)]
del dataset_encoded

print('Splitting data into training/validation sets')
val_size = 0.1
seed = 0
from sklearn import model_selection
X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=val_size, random_state=seed)
del X
del Y

##########################
### ENSEMBLE LEARNING ###
##########################

# ensemble learning using xgboost models only
# outputs submission.csv ready to be submitted to kaggle

if ensembleLearnerXGBoost:

    from xgboost import XGBRegressor
    n_list = numpy.array([50,150,250,350,450])

    print('\n\nEnsemble learning with XGBoost models with n_estimators = ' + str(n_list))

    X = numpy.concatenate((X_train,X_val),axis=0)
    Y = numpy.concatenate((Y_train,Y_val),axis=0)
    del X_train
    del X_val
    del Y_train
    del Y_val

    Y_val_preds = numpy.full((len(n_list),dataset_submission_encoded.shape[0]),numpy.nan,dtype=float)
    modelnameseval = []

    for index in range(len(n_list)):

        n_estimators = n_list[index]
        print('\nTraining for n_estimators = ' + str(n_list[index]))

        print('Fitting model')
        model = XGBRegressor(n_estimators=n_estimators,
                             num_boost_round=200,
                             gamma=0.2,
                             max_depth=8,
                             min_child_weight=6,
                             colsample_bytree=0.6,
                             subsample=0.9,
                             eta=0.07)
        # see https://github.com/dnkirill/allstate_capstone/blob/master/part2_xgboost.ipynb
        model.fit(X,Y)

        print('Making predictions')
        predictions = model.predict(dataset_submission_encoded)
        predictions = ce4041.inverseTransformData(predictions)

        print('Collecting values for ensemble learning')
        Y_val_preds[index] = predictions
        modelnameseval = numpy.append(modelnameseval,'XGBoost' + str(n_list[index]))
        
        print('Writing to file')
        filename = "XGBoost" + str(n_list[index]) + ".csv"
        with open(filename, "w") as subfile:
            subfile.write("id,loss\n")
            for i, pred in enumerate(list(predictions)):
                subfile.write("%s,%s\n"%(dataset_submission_ID[i],pred))

    print('\n\nAveraging')
    Y_preds = numpy.full(Y_val_preds.shape[1],0,dtype=float)
    weight = (float)(1.0/(float)(Y_val_preds.shape[0]))
    for dataindex in range(Y_val_preds.shape[1]):
        for modelindex in range(Y_val_preds.shape[0]):
            Y_preds[dataindex] += weight*Y_val_preds[modelindex][dataindex]

    print('Writing to file')
    with open("submission.csv", "w") as subfile:
        subfile.write("id,loss\n")
        for i, pred in enumerate(list(Y_preds)):
            subfile.write("%s,%s\n"%(dataset_submission_ID[i],pred))

# general ensemble learning
# prints out the different models, their optimal parameters and corresponding lowest maes,
# and maes of the resultant ensemble learners
# recommended NOT to use this - don't include any of the models except XGBoost
# because they give much worse MAEs than XGBoost variants

if ensembleLearner:

    returntype='val'
    index = 0
    
    models =[
        ['LinearReg',		EvalLinearReg,		ce4041.LinearReg,	ParaLinearReg],
        ['RidgeReg',		EvalRidgeReg,		ce4041.RidgeReg,	ParaRidgeReg],
        ['LASSOLinearReg', 	EvalLASSOLinearReg,	ce4041.LASSOLinearReg,	ParaLASSOLinearReg],
        ['ElasticNetReg',	EvalElasticNetReg,	ce4041.ElasticNetReg,	ParaElasticNetReg],
        ['KNN',			EvalKNN,		ce4041.KNN,		ParaKNN],
        ['CART',		EvalCART,		ce4041.CART,		ParaCART],
        ['SVM',			EvalSVM,		ce4041.SVM,		ParaSVM],
        ['BaggedDT',		EvalBaggedDT,		ce4041.BaggedDT,	ParaBaggedDT],
        ['RandomForest',	EvalRandomForest,	ce4041.RandomForest,	ParaRandomForest],
        ['ExtraTrees',		EvalExtraTrees,		ce4041.ExtraTrees,	ParaExtraTrees],
        ['AdaBoost',		EvalAdaBoost,		ce4041.AdaBoost,	ParaAdaBoost],
        ['SGBoost',		EvalSGBoost,		ce4041.SGBoost,		ParaSGBoost],
        ['MLP',			EvalMLP,		ce4041.MLP,		ParaMLP],
        ['XGBoost',		EvalXGBoost,		ce4041.XGBoost,		ParaXGBoost],
    ]

##    models = [
##        ['XGBoost50',   True,   ce4041.XGBoost, numpy.array([50])],
##        ['XGBoost150',   True,   ce4041.XGBoost, numpy.array([150])],
##        ['XGBoost250',   True,   ce4041.XGBoost, numpy.array([250])],
##        ['XGBoost350',   True,   ce4041.XGBoost, numpy.array([350])],
##        ['XGBoost450',   True,   ce4041.XGBoost, numpy.array([450])],
##        ['XGBoost550',   True,   ce4041.XGBoost, numpy.array([550])],
##        ['XGBoost650',   True,   ce4041.XGBoost, numpy.array([650])],
##        ['XGBoost750',   True,   ce4041.XGBoost, numpy.array([750])],
##        ['XGBoost850',   True,   ce4041.XGBoost, numpy.array([850])],
##        ['XGBoost950',   True,   ce4041.XGBoost, numpy.array([950])],
##        ['XGBoost1050',   True,   ce4041.XGBoost, numpy.array([1050])],        
##        ]

    # initialize arrays that store performance
    Y_val_preds = numpy.full((len(models),X_val.shape[0]),numpy.nan,dtype=float)
    Y_train_preds = numpy.full((len(models),X_train.shape[0]),numpy.nan,dtype=float)
    minparas = numpy.full(len(models),numpy.nan,dtype=float)
    minmaes = numpy.full(len(models),numpy.nan,dtype=float)
    modelnameseval = []
    
    for name, enabled, function, parameters in models:
        if enabled:
            Y_train_preds[index], Y_val_preds[index], minparas[index], minmaes[index] = function(X_train, Y_train, X_val, Y_val, parameters, returntype=returntype)
            modelnameseval = numpy.append(modelnameseval,name)
            filename = str(index+1) + ' ' + str(name.strip()) + ' ' + str(minparas[index]) + ' ' + str(numpy.around(minmaes[index],3)) + ".csv"
            ce4041.writeToCSV(Y_val_preds[index],filename)
            index += 1
            
    # remove nan values so only enabled models remain
    Y_val_preds = Y_val_preds[~numpy.isnan(Y_val_preds).any(axis=1)]
    Y_train_preds = Y_train_preds[~numpy.isnan(Y_train_preds).any(axis=1)]
    minparas = minparas[~numpy.isnan(minparas)]
    minmaes = minmaes[~numpy.isnan(minmaes)]

    print('\n\nOptimal Parameters & Min MAE')
    print('Name          MinPara\tMinMAE')
    for i in range(len(modelnameseval)):
        print(modelnameseval[i],end='\t')
        print(minparas[i],end='\t')
        print(minmaes[i],end='\t')
        print()

    # TODO alternate methods for determining cutoff
    # generate array of cutoff values that will include more and more models
    # ensemble learner will start from  2 models with lowest MAEs,
    #                                   3 models with lowest MAEs,
    #                                   ... max num of models

    minmaessorted = numpy.sort(minmaes)
    minmaessorted = minmaessorted[~numpy.isnan(minmaessorted)] 
    minmaessorted = minmaessorted[~numpy.isinf(minmaessorted)]
    cutoffmae = []

    cutoffs = numpy.empty(0,dtype=int)
    for i in range(1,len(minmaessorted)):
        cutoffs = numpy.append(cutoffs,int(minmaessorted[i])+1)

    for cutoff in cutoffs:

        modelnamesens = []
        minmaesens = []
        Y_preds_collection = []
        weightsens = []

        if returntype == 'none' or returntype == 'train':
            print('Y_val_preds invalid!')
            sys.exit()

        # collect model names, maes and predicted values
        # for qualifying models
        for i in range(len(modelnameseval)):
            if minmaes[i] < cutoff:
                modelnamesens = numpy.append(modelnamesens,modelnameseval[i])
                minmaesens = numpy.append(minmaesens,minmaes[i])
                Y_preds_collection = numpy.append(Y_preds_collection,Y_val_preds[i])

        print('##########################')
        print('Cutoff = ' + str(cutoff))

        # TODO different weights based on MAE
        weight = numpy.full(len(modelnamesens),(float)(1.0/len(modelnamesens)),dtype=float)

        # print selected models
        print('\nSelected Models')
        print('Name\tMAE\tweight')
        for i in range(len(modelnamesens)):
            print(modelnamesens[i],end='\t')
            print(numpy.around(minmaesens[i],3),end='\t')
            print(weight[i],end='\t')
            print()

        # get ensemble predicted values
        Y_preds_ensemble = numpy.zeros(Y_val.shape,dtype=float)
        for dataindex in range(len(Y_val)):
            for modelindex in range(len(modelnamesens)):
                Y_preds_ensemble[dataindex] += weight[modelindex]*Y_preds_collection[(dataindex+(len(Y_val))*(modelindex))]

        # calculate MAE for ensemble
        finalresult = ce4041.getResult(Y_val, Y_preds_ensemble)
        cutoffmae = numpy.append(cutoffmae,finalresult)
        print('\nMAE\t' + str(numpy.around(finalresult,3)))

    print('##########################')
    print('MAEs = ' + str(cutoffmae))
    print('Optimal cutoff = ' + str(cutoffs[numpy.argmin(cutoffmae)]) + ' with resultant MAE of ensemble learner = '  + str(numpy.around(cutoffmae[numpy.argmin(cutoffmae)],3)))

# make predictions using 1 model only

if makePredictions:
    print('\nMaking predictions...',end='')
    X = numpy.concatenate((X_train,X_val),axis=0)
    Y = numpy.concatenate((Y_train,Y_val),axis=0)

    n_estimators=100

    # see https://github.com/dnkirill/allstate_capstone/blob/master/part2_xgboost.ipynb
    model = XGBRegressor(num_boost_round=200, gamma=0.2, max_depth=8, min_child_weight=6, colsample_bytree=0.6, subsample=0.9, eta=0.07)
    model.fit(X,Y)

    # Make predictions using the best model
    predictions = ce4041.inverseTransformData(model.predict(dataset_submission_encoded))
    print('\tFinished.')
    
    # Write submissions to output file in the correct format
    print('Writing to file...',end='')
    with open("submission.csv", "w") as subfile:
        subfile.write("id,loss\n")
        for i, pred in enumerate(list(predictions)):
            subfile.write("%s,%s\n"%(dataset_submission_ID[i],pred))
    print('\tFinished.')
