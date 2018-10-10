# coding=utf-8
import time
import pandas as pd
import matplotlib.pyplot as plt
import DFFunctions as DFFuncs
import FitFunctions as FitFuncs
import plotAnalysis as plotMethods
import datetime as dt
import numpy as np

from sklearn import datasets, linear_model
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


###################################################################
###################################################################
###################################################################
###################################################################

class DataAnalysis():

###################################################################
###################################################################

    def __init__(self, dataDir, plotsDir):

        self.dataDir = dataDir
        self.plotsDir = plotsDir

        self.figWidth = 15
        self.figHeight = 8
        self.linewidth = 2

        self.tiltBool = True
        self.rotation = 30

        self.plotData = plotMethods.plotAnalysis(plotsDir=self.plotsDir)
        self.DFFunc = DFFuncs.DFFunctions()
        self.FitFunc = FitFuncs.FitFunctions()

        return

###################################################################
###################################################################
    
    def runAnalysis(self):

        self.landForest_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_AG.LND.FRST.ZS_DS2_en_csv_v2_10052112.csv', skiprow=4)
        self.atmosphereCO2_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_EN.ATM.CO2E.KT_DS2_en_csv_v2_10051706.csv', skiprow=4)
        self.GDP_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_10051799.csv', skiprow=4)
        self.populationTotal_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_SP.POP.TOTL_DS2_en_csv_v2_10058048.csv', skiprow=4)
        self.populationUrban_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_10034507.csv', skiprow=4)

        # print(self.landForest_DF.tail(5))
        # print(self.atmosphereCO2_DF.tail(5))
        # print(self.GDP_DF.tail(5))
        # print(self.populationTotal_DF.tail(5))
        # print(self.populationUrban_DF.tail(5))

        # print(self.populationUrban_DF.shape)

        dfList = [self.landForest_DF, self.atmosphereCO2_DF, self.GDP_DF, self.populationTotal_DF, self.populationUrban_DF]
        dfColumnHeaders = ['landForest', 'atmosphereCO2', 'GDP', 'populationTotal', 'populationUrban']

        trainSetupDF = pd.DataFrame({'Country':self.landForest_DF['Country Name'], 'CountryType':self.landForest_DF['CountryType']})
        testSetupDF = pd.DataFrame({'Country':self.landForest_DF['Country Name'], 'CountryType':self.landForest_DF['CountryType']})

        for i in range(0,len(dfList)):
            tempDF = dfList[i]
            # Pick year with data in every variable, particularly atmosphereCO2
            trainSetupDF[dfColumnHeaders[i]] = tempDF['2013']
            testSetupDF[dfColumnHeaders[i]] = tempDF['2014']

        trainDF = self.DFFunc.setupAnalysisDF(trainSetupDF)
        # trainDF = self.DFFunc.setupAnalysisDF(trainSetupDF, countryType='Group')
        print(trainDF.shape)
        print(trainDF.tail(5))

        testDF = self.DFFunc.setupAnalysisDF(testSetupDF)

        train_predictors = trainDF.drop(['atmosphereCO2', 'CountryType', 'Country'], axis=1).copy()
        train_target = pd.DataFrame({'atmosphereCO2':trainDF['atmosphereCO2']})

        test_predictors = testDF.drop(['atmosphereCO2', 'CountryType', 'Country'], axis=1).copy()
        test_target = pd.DataFrame({'atmosphereCO2':testDF['atmosphereCO2']})

        ### Linear Regression

        # with sklearn
        # regr = linear_model.LinearRegression()
        # regr.fit(X, Y)

        # print('Intercept: \n', regr.intercept_)
        # print('Coefficients: \n', regr.coef_)

        # with statsmodels
        train_predictors = sm.add_constant(train_predictors) # adding a constant
        test_predictors = sm.add_constant(test_predictors) # adding a constant
        
        LR_model = sm.OLS(train_target, train_predictors).fit()
        LR_train_predictions = LR_model.predict(train_predictors)

        resultsTrainDF = train_target.copy()
        resultsTrainDF['LR_train_Predictions'] = LR_train_predictions
        resultsTrainDF['LR_train_Residue'] = resultsTrainDF['LR_train_Predictions'] - resultsTrainDF['atmosphereCO2']

        LR_test_predictions = LR_model.predict(test_predictors)

        resultsTestDF = test_target.copy()
        resultsTestDF['LR_test_Predictions'] = LR_test_predictions
        resultsTestDF['LR_test_Residue'] = resultsTestDF['LR_test_Predictions'] - resultsTestDF['atmosphereCO2']

        LR_print_model = LR_model.summary()
        print(LR_print_model)

        ### Random Forest

        # self.FitRandomForest(train_predictors, test_predictors, train_target, test_target)

        # rf = best_grid

        # ### Neural Network

        # self.FitNeuralNetwork(train_predictors, test_predictors, train_target, test_target)

        # ### Ensemble Decision Trees

        # ### XGBoost

        # # Use the forest's predict method on the test data
        # RF_train_predictions = rf.predict(train_predictors)
        # RF_test_predictions = rf.predict(test_predictors)

        # resultsTrainDF['RF_train_Predictions'] = RF_train_predictions
        # resultsTrainDF['RF_train_Residue'] = resultsTrainDF['RF_train_Predictions'] - resultsTrainDF['atmosphereCO2']


        # resultsTestDF['RF_test_Predictions'] = RF_test_predictions
        # resultsTestDF['RF_test_Residue'] = resultsTestDF['RF_test_Predictions'] - resultsTestDF['atmosphereCO2']

        # # # Calculate the absolute errors
        # # errors = abs(predictions - test_labels)
        # LR_errors = abs(resultsTestDF['LR_test_Residue'])
        # RF_errors = abs(resultsTestDF['RF_test_Residue'])
        # # # Print out the mean absolute error (mae)
        # print('Mean Absolute Error (Linear Regression):', round(abs(resultsTestDF['LR_test_Residue']).mean(), 2), '.')
        # print('Mean Absolute Error (Random Forest):', round(abs(resultsTestDF['RF_test_Residue']).mean(), 2), '.')
        # # Mean Absolute Error: 3.83 degrees.
        # # # Calculate mean absolute percentage error (MAPE)
        # # mape = 100 * (errors / test_labels)
        # LR_mape = 100 * (LR_errors / resultsTestDF['atmosphereCO2'])
        # RF_mape = 100 * (RF_errors / resultsTestDF['atmosphereCO2'])
        # # # Calculate and display accuracy
        # # accuracy = 100 - np.mean(mape)
        # LR_accuracy = 100 - LR_mape.mean()
        # RF_accuracy = 100 - RF_mape.mean()
        # print('Accuracy (Linear Regression):', round(LR_accuracy, 2), '%.')
        # print('Accuracy (Random Forest):', round(RF_accuracy, 2), '%.')
        # Accuracy: 93.99 %.

        # print(len(self.landForest_DF['Country Name'].unique()))
        # print(len(self.atmosphereCO2_DF['Country Name'].unique()))
        # print(len(self.GDP_DF['Country Name'].unique()))
        # print(len(self.populationTotal_DF['Country Name'].unique()))
        # print(len(self.populationUrban_DF['Country Name'].unique()))

        # print(self.landForest_DF['2017'])

        # self.plotData.plotPairsDF(combineDF)

        # self.plotData.plotGraph(combineDF['landForest'], combineDF['atmosphereCO2'], title="atmosphereCO2 VS landForest", xlabel="landForest", ylabel="atmosphereCO2", legendLabel1="Countries", outputFileName="atmosphereCO2_VS_landForest.png", time=False)

        # self.plotData.plotTargetGraph(trainDF.index.values, trainDF['atmosphereCO2'], title="atmosphereCO2 2014", xlabel="Country", ylabel="atmosphereCO2", legendLabel1="Indicies of countries", outputFileName="atmosphereCO2_2014.png", time=False, tilt=False, xTickRotation=30)

        # self.plotData.plotResultGraph(train_target.index.values, [train_target['atmosphereCO2'], train_target['Predictions']], title="atmosphereCO2 2014", xlabel="Country", ylabel="atmosphereCO2", legendLabel=["Data", "Predictions"], outputFileName="atmosphereCO2_results.png", tilt=False, xTickRotation=30)

        # self.plotData.plotResultGraph(resultsTrainDF.index.values, [resultsTrainDF['LR_train_Residue'], resultsTrainDF['RF_train_Residue']], title="atmosphereCO2 2014", xlabel="Country", ylabel="atmosphereCO2", legendLabel=["LR_train_Residue", "RF_train_Residue"], outputFileName="atmosphereCO2_train_residue.png", tilt=False, xTickRotation=30)

        # self.plotData.plotResultGraph(resultsTestDF.index.values, [resultsTestDF['LR_test_Residue'], resultsTestDF['RF_test_Residue']], title="atmosphereCO2 2014", xlabel="Country", ylabel="atmosphereCO2", legendLabel=["LR_test_Residue", "RF_test_Residue"], outputFileName="atmosphereCO2_test_residue.png", tilt=False, xTickRotation=30)

        # self.plotData.plotBarGraph(self.GDP_DF['Country Name'], self.GDP_DF['2017'], title="GDP 2017", xlabel="Country", ylabel="GDP", legendLabel1="GDP 2017", outputFileName="GDP_2017.png", tilt=self.tiltBool, xTickRotation=self.rotation)

        landForest_DF = self.landForest_DF[self.landForest_DF['CountryType']=='Country'].copy()
        atmosphereCO2_DF = self.atmosphereCO2_DF[self.atmosphereCO2_DF['CountryType']=='Country'].copy()
        GDP_DF = self.GDP_DF[self.GDP_DF['CountryType']=='Country'].copy()
        populationTotal_DF = self.populationTotal_DF[self.populationTotal_DF['CountryType']=='Country'].copy()
        populationUrban_DF = self.populationUrban_DF[self.populationUrban_DF['CountryType']=='Country'].copy()

        self.plotData.plotParallelCoordinateGraph(trainDF, title="Parallel Coordinates Graph", xlabel="Quantities", ylabel="Countries")

        ###################################################################

        # self.plotData.plotBarGraph(landForest_DF['Country Name'], landForest_DF['2013'], title="Land Forest 2013", xlabel="Country", ylabel="Land Forest", legendLabel1="Land Forest 2013", outputFileName="LandForest_2013.png", tilt=self.tiltBool, xTickRotation=self.rotation)

        # self.plotData.plotBarGraph(atmosphereCO2_DF['Country Name'], atmosphereCO2_DF['2013'], title="CO2 2013", xlabel="Country", ylabel="CO2", legendLabel1="CO2 2013", outputFileName="CO2_2013.png", tilt=self.tiltBool, xTickRotation=self.rotation)

        # self.plotData.plotBarGraph(GDP_DF['Country Name'], GDP_DF['2013'], title="GDP 2013", xlabel="Country", ylabel="GDP", legendLabel1="GDP 2013", outputFileName="GDP_2013.png", tilt=self.tiltBool, xTickRotation=self.rotation)

        # self.plotData.plotBarGraph(populationTotal_DF['Country Name'], populationTotal_DF['2013'], title="Total Population 2013", xlabel="Country", ylabel="Total Population", legendLabel1="Total population 2013", outputFileName="populationTotal_2013.png", tilt=self.tiltBool, xTickRotation=self.rotation)

        # self.plotData.plotBarGraph(populationUrban_DF['Country Name'], populationUrban_DF['2013'], title="Urban Population 2013", xlabel="Country", ylabel="Urban Population", legendLabel1="Urban Population 2013", outputFileName="populationUrban_2013.png", tilt=self.tiltBool, xTickRotation=self.rotation)

        ###################################################################

        # self.plotData.plotBarGraph(landForest_DF['Country Name'], landForest_DF['2013'], title="Land Forest 2013", xlabel="Country", ylabel="Land Forest", legendLabel1="Land Forest 2013", outputFileName="LandForest_2013_bottom.png", tilt=self.tiltBool, xTickRotation=self.rotation, bottom=True)

        # self.plotData.plotBarGraph(atmosphereCO2_DF['Country Name'], atmosphereCO2_DF['2013'], title="CO2 2013", xlabel="Country", ylabel="CO2", legendLabel1="CO2 2013", outputFileName="CO2_2013_bottom.png", tilt=self.tiltBool, xTickRotation=self.rotation, bottom=True)

        # self.plotData.plotBarGraph(GDP_DF['Country Name'], GDP_DF['2013'], title="GDP 2013", xlabel="Country", ylabel="GDP", legendLabel1="GDP 2013", outputFileName="GDP_2013_bottom.png", tilt=self.tiltBool, xTickRotation=self.rotation, bottom=True)

        # self.plotData.plotBarGraph(populationTotal_DF['Country Name'], populationTotal_DF['2013'], title="Total Population 2013", xlabel="Country", ylabel="Total Population", legendLabel1="Total population 2013", outputFileName="populationTotal_2013_bottom.png", tilt=self.tiltBool, xTickRotation=self.rotation, bottom=True)

        # self.plotData.plotBarGraph(populationUrban_DF['Country Name'], populationUrban_DF['2013'], title="Urban Population 2013", xlabel="Country", ylabel="Urban Population", legendLabel1="Urban Population 2013", outputFileName="populationUrban_2013_bottom.png", tilt=self.tiltBool, xTickRotation=self.rotation, bottom=True)

        plt.show()

        return

###################################################################
###################################################################

    def FitNeuralNetwork(self, train_predictors, test_predictors, train_target, test_target):

        scaler = StandardScaler()
        # Fit only to the training data
        scaler.fit(train_predictors)

        # Now apply the transformations to the data:
        # train_predictors = scaler.transform(train_predictors)
        # test_predictors = scaler.transform(test_predictors)

        # mlp = MLPRegressor(hidden_layer_sizes=(30,30,30))
        # mlp = MLPRegressor()
        # mlp.fit(train_predictors,train_target.values.ravel())

        mlp = self.FitFunc.FitNeuralNetwork(train_predictors=train_predictors, train_target=train_target, searchType="Random")

        NN_accuracy = self.FitFunc.evaluate(mlp, test_predictors, test_target.values.ravel())

        return

###################################################################
###################################################################

    def FitRandomForest(self, train_predictors, test_predictors, train_target, test_target):

        ## Fine tune hyperparameters with Randomized Search

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']

        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)

        # Minimum number of samples required to split a node
        min_samples_split = [2, 4, 6, 8, 10, 12]

        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 3, 4, 5]

        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        rf_random = self.FitFunc.FitRandomForest(paramGrid=random_grid, train_predictors=train_predictors, train_target=train_target)

        base_model = self.FitFunc.FitRandomForest(train_predictors=train_predictors, train_target=train_target)
        base_accuracy = self.FitFunc.evaluate(base_model, test_predictors, test_target.values.ravel())

        best_random = rf_random.best_estimator_
        random_accuracy = self.FitFunc.evaluate(best_random, test_predictors, test_target.values.ravel())

        print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

        ## Fine tune hyperparameters with Grid Search

        param_grid = {
            'n_estimators': [400, 600, 800, 1000, 1200],
            'max_features': ['auto'],
            'min_samples_split': [8, 10, 12],
            'min_samples_leaf': [1, 2, 3],
            'max_depth': [60, 70, 80, 90, 100],
            'bootstrap': [False]
        }

        grid_search = self.FitFunc.FitRandomForest(paramGrid=param_grid, train_predictors=train_predictors, train_target=train_target, searchType="Grid")
        best_grid = grid_search.best_estimator_
        grid_accuracy = self.FitFunc.evaluate(best_grid, test_predictors, test_target.values.ravel())

        print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

        return

###################################################################
###################################################################



if __name__ == "__main__":

    startTime = time.time()
    
    dataDir = '../Data/'
    plotsDir = '../Plots/'
        
    Analysis_Object = DataAnalysis(dataDir, plotsDir)
    
    Analysis_Object.runAnalysis()
    
    endTime = time.time()
    
    print ("Time elapsed: " + repr(endTime-startTime))