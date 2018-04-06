## Ensemble learning

from __future__ import print_function
print('Importing libraries')
import numpy
import xgboost as xgb
from xgboost import XGBRegressor
import sys
import ce4041
from sklearn import model_selection

fulldataset = True
n_list = numpy.arange(start=200,stop=1800,step=150)

dataset, dataset_submission, dataset_submission_ID = ce4041.readData(fulldataset=fulldataset)
print('\nTransforming outputs using log(loss+' + str(ce4041.shift) + ')')
dataset["loss"] = ce4041.transformData(dataset["loss"])
print('One-hot encoding of categorical inputs')
dataset_encoded, dataset_submission_encoded = ce4041.oneHotEncoding(dataset, dataset_submission, oneHotEncode=False)

print('Splitting data into inputs/outputs')
X = dataset_encoded[:,0:(dataset_encoded.shape[1]-1)]
Y = dataset_encoded[:,(dataset_encoded.shape[1]-1)]
del dataset, dataset_submission, dataset_encoded

print('\n\nEnsemble learning with XGBoost models with n_estimators = ' + str(n_list))

Y_val_preds = numpy.full((len(n_list),dataset_submission_encoded.shape[0]),numpy.nan,dtype=float)
modelnameseval = []

for index in range(len(n_list)):

    n_estimators = numpy.array([n_list[index]])
    print('\nTraining for n_estimators = ' + str(n_list[index]))

    print('Fitting model')
##    model = XGBRegressor(n_estimators=n_estimators,
##                         num_boost_round=200,
##                         gamma=0.2,
##                         max_depth=8,
##                         min_child_weight=6,
##                         colsample_bytree=0.6,
##                         subsample=0.9,
##                         eta=0.07)
##    # see https://github.com/dnkirill/allstate_capstone/blob/master/part2_xgboost.ipynb
    model = XGBRegressor(n_estimators=n_estimators,
                            min_child_weight= 1,
                            eta= 0.01,
                            colsample_bytree= 0.5,
                            max_depth= 12,
                            subsample= 0.8,
                            alpha= 1,
                            gamma= 1,
                            silent= 1,
                            verbose_eval= True,
                            seed=0)
    # see https://www.kaggle.com/iglovikov/xgb-1114/code
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
