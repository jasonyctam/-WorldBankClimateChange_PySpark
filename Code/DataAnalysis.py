# coding=utf-8
import time
import pandas as pd
import matplotlib.pyplot as plt
import DFFunctions as DFFuncs
import FitFunctions as FitFuncs
import plotAnalysis as plotMethods
import datetime as dt
import numpy as np

import pyspark
import statsmodels.api as sm

from pyspark import SparkContext # For converting pandas dataframe to pyspark dataframe
from pyspark.sql import SQLContext # For converting pandas dataframe to pyspark dataframe

from pyspark.ml.feature import VectorAssembler # For preparing a DF in a format for ML functions input
from pyspark.ml.regression import LinearRegression # For Linear Regression with PySpark
from pyspark.ml.evaluation import RegressionEvaluator # For evaluating regression model predictions
from pyspark.ml.regression import DecisionTreeRegressor # For Decision Tree Regression with PySpark
from pyspark.ml.regression import GBTRegressor # For Gradient Boosted Tree Regression with PySpark
from pyspark.ml import Pipeline # For using PySpark pipelines

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

        self.sc = SparkContext("local", "App Name")
        self.sql = SQLContext(self.sc)

        return

###################################################################
###################################################################
    
    def runAnalysis(self):

        self.landForest_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_AG.LND.FRST.ZS_DS2_en_csv_v2_10052112.csv', skiprow=4)
        self.atmosphereCO2_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_EN.ATM.CO2E.KT_DS2_en_csv_v2_10051706.csv', skiprow=4)
        self.GDP_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_10051799.csv', skiprow=4)
        self.populationTotal_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_SP.POP.TOTL_DS2_en_csv_v2_10058048.csv', skiprow=4)
        self.populationUrban_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_10034507.csv', skiprow=4)

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

        testDF = self.DFFunc.setupAnalysisDF(testSetupDF)

        # train_predictors_DF = trainDF.drop(['atmosphereCO2', 'CountryType', 'Country'], axis=1).copy()
        # train_target_DF = pd.DataFrame({'atmosphereCO2':trainDF['atmosphereCO2']})

        # test_predictors_DF = testDF.drop(['atmosphereCO2', 'CountryType', 'Country'], axis=1).copy()
        # test_target_DF = pd.DataFrame({'atmosphereCO2':testDF['atmosphereCO2']})

        # train_predictors = self.sql.createDataFrame(train_predictors_DF)
        # train_target = self.sql.createDataFrame(train_target_DF)

        # test_predictors = self.sql.createDataFrame(test_predictors_DF)
        # test_target = self.sql.createDataFrame(test_target_DF)

        # train_predictors.show()
        # train_target.show()
        # test_predictors.show()
        # test_target.show()

        # Use Vector Assembler to prepare a DF for ML functions

        train_DF = self.sql.createDataFrame(trainDF)
        test_DF = self.sql.createDataFrame(testDF)

        vectorAssembler = VectorAssembler(inputCols = ['landForest', 'populationTotal', 'GDP', 'populationUrban'], outputCol = 'features')
        trainMLDF = vectorAssembler.transform(train_DF)
        trainMLDF = trainMLDF.select(['features', 'atmosphereCO2'])
        # trainMLDF.show(3)

        testMLDF = vectorAssembler.transform(test_DF)
        testMLDF = testMLDF.select(['features', 'atmosphereCO2'])
        # trainMLDF.show(3)

        ### Linear Regression

        lr_DF = self.FitLinearRegression(trainMLDF, testMLDF)
        print(lr_DF.head(5))

        ## Decision Tree

        # Fit with default parameters
        dt_DF = self.RunDecisionTree(trainMLDF, testMLDF)
        print(dt_DF.head(5))

        ## Gradient Boosted Trees

        gbt_DF = self.RunGradientBoostedTree(trainMLDF, testMLDF)
        print(gbt_DF.head(5))

        resultsDF = lr_DF.copy()
        resultsDF['LR_Res'] = lr_DF['prediction'] - lr_DF['atmosphereCO2']
        resultsDF = resultsDF.drop(['prediction'], axis=1).copy()
        resultsDF['DT_Res'] = dt_DF['prediction'] - dt_DF['atmosphereCO2']
        resultsDF['GBT_Res'] = gbt_DF['prediction'] - gbt_DF['atmosphereCO2']


        self.plotData.plotResultGraph(resultsDF.index.values, [resultsDF['LR_Res'], resultsDF['DT_Res'], resultsDF['GBT_Res']], title="atmosphereCO2 2014", xlabel="Country", ylabel="atmosphereCO2", legendLabel=["LR_test_Residue", "DT_test_Residue", "GBT_test_Residue"], outputFileName="atmosphereCO2_test_residue.png", tilt=False, xTickRotation=30)


        # self.plotData.plotGraph(combineDF['landForest'], combineDF['atmosphereCO2'], title="atmosphereCO2 VS landForest", xlabel="landForest", ylabel="atmosphereCO2", legendLabel1="Countries", outputFileName="atmosphereCO2_VS_landForest.png", time=False)

        # self.plotData.plotTargetGraph(trainDF.index.values, trainDF['atmosphereCO2'], title="atmosphereCO2 2014", xlabel="Country", ylabel="atmosphereCO2", legendLabel1="Indicies of countries", outputFileName="atmosphereCO2_2014.png", time=False, tilt=False, xTickRotation=30)

        # self.plotData.plotResultGraph(train_target.index.values, [train_target['atmosphereCO2'], train_target['Predictions']], title="atmosphereCO2 2014", xlabel="Country", ylabel="atmosphereCO2", legendLabel=["Data", "Predictions"], outputFileName="atmosphereCO2_results.png", tilt=False, xTickRotation=30)

        # self.plotData.plotResultGraph(resultsTrainDF.index.values, [resultsTrainDF['LR_train_Residue'], resultsTrainDF['RF_train_Residue']], title="atmosphereCO2 2014", xlabel="Country", ylabel="atmosphereCO2", legendLabel=["LR_train_Residue", "RF_train_Residue"], outputFileName="atmosphereCO2_train_residue.png", tilt=False, xTickRotation=30)

        # self.plotData.plotResultGraph(resultsTestDF.index.values, [resultsTestDF['LR_test_Residue'], resultsTestDF['RF_test_Residue']], title="atmosphereCO2 2014", xlabel="Country", ylabel="atmosphereCO2", legendLabel=["LR_test_Residue", "RF_test_Residue"], outputFileName="atmosphereCO2_test_residue.png", tilt=False, xTickRotation=30)

        # self.plotData.plotBarGraph(self.GDP_DF['Country Name'], self.GDP_DF['2017'], title="GDP 2017", xlabel="Country", ylabel="GDP", legendLabel1="GDP 2017", outputFileName="GDP_2017.png", tilt=self.tiltBool, xTickRotation=self.rotation)

        # landForest_DF = self.landForest_DF[self.landForest_DF['CountryType']=='Country'].copy()
        # atmosphereCO2_DF = self.atmosphereCO2_DF[self.atmosphereCO2_DF['CountryType']=='Country'].copy()
        # GDP_DF = self.GDP_DF[self.GDP_DF['CountryType']=='Country'].copy()
        # populationTotal_DF = self.populationTotal_DF[self.populationTotal_DF['CountryType']=='Country'].copy()
        # populationUrban_DF = self.populationUrban_DF[self.populationUrban_DF['CountryType']=='Country'].copy()

        # self.plotData.plotParallelCoordinateGraph(trainDF, title="Parallel Coordinates Graph", xlabel="Quantities", ylabel="Countries")

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

        # plt.show()

        return

###################################################################
###################################################################

    def FitLinearRegression(self, trainMLDF, testMLDF):

        print("Running Linear Regression.....")

        lr = LinearRegression(featuresCol = 'features', labelCol='atmosphereCO2', maxIter=100, regParam=0.3, elasticNetParam=0.8)
        lr_model = lr.fit(trainMLDF)
        print("Coefficients: " + str(lr_model.coefficients))
        print("Intercept: " + str(lr_model.intercept))

        trainingSummary = lr_model.summary
        print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
        print("r2: %f" % trainingSummary.r2)

        # trainMLDF.describe().show()

        lr_predictions = lr_model.transform(testMLDF)
        lr_predictions.select("prediction","atmosphereCO2","features").show(5)
        lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="atmosphereCO2",metricName="r2")
        print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

        test_result = lr_model.evaluate(testMLDF)
        print("Linear Regression Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)

        print("numIterations: %d" % trainingSummary.totalIterations)
        print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
        # trainingSummary.residuals.show()

        # predictions = lr_model.transform(testMLDF)
        # predictions.select("prediction","atmosphereCO2","features").show()

        return lr_predictions.toPandas()

###################################################################
###################################################################

    def RunDecisionTree(self, trainMLDF, testMLDF):

        deafultDF = self.FitDecisionTree(trainMLDF, testMLDF)

        # Best parameters after running the following
        # 1	20	36	37946.8129595073
        # [minInstancesPerNode, maxDepth, maxBins, dt_rmse]

        dt_maxDepth_array = [20]
        dt_maxBins_array = [36]
        dt_minInstancesPerNode_array = [1]

        # Swap for these to running the hyper-parameter search again
        # dt_maxDepth_array = [10,20,30]
        # dt_maxBins_array = [32,34,36]
        # dt_minInstancesPerNode_array = [1,2,3]

        dt_results_array = [] # minInstancesPerNode, maxDepth, maxBins, rmse

        for i in range(0, len(dt_maxDepth_array)):
            for j in range(0, len(dt_maxBins_array)):
                for k in range(0, len(dt_minInstancesPerNode_array)):
                    dt_maxDepth = dt_maxDepth_array[i]
                    dt_maxBins = dt_maxBins_array[j]
                    dt_minInstancesPerNode = dt_minInstancesPerNode_array[k]

                    dt_params = {
                        "maxDepth":dt_maxDepth,
                        "maxBins":dt_maxBins,
                        "minInstancesPerNode":dt_minInstancesPerNode
                    }

                    dt_results, dt_DF = self.FitDecisionTree(trainMLDF, testMLDF, dt_params)

                    dt_results_array.append(dt_results)

                    del dt_maxDepth
                    del dt_maxBins
                    del dt_minInstancesPerNode
                    del dt_params

        for i in range(0,len(dt_results_array)):
            print(dt_results_array[i])

        return dt_DF


###################################################################
###################################################################

    def FitDecisionTree(self, trainMLDF, testMLDF, params={}):

        print("Running Decision Tree.....")

        if bool(params):
            print("Fitting with maxDepth = " + str(params["maxDepth"]) + ", maxBins = " + str(params["maxBins"]) + ", minInstancesPerNode = " + str(params["minInstancesPerNode"]) + " ...")
            dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'atmosphereCO2', maxDepth=params["maxDepth"], maxBins = params["maxBins"], minInstancesPerNode = params["minInstancesPerNode"])
        else:
            print("Fitting with default parameters...")
            dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'atmosphereCO2')

        # dt_paramGrid = ParamGridBuilder()\
        #     .addGrid(dt.maxDepth, [10,20,30]) \
        #     .build()

        dt_pipeline = Pipeline(stages=[dt])
        dt_evaluator = RegressionEvaluator(labelCol="atmosphereCO2", predictionCol="prediction", metricName="rmse")

        dt_model = dt_pipeline.fit(trainMLDF)
        dt_predictions = dt_model.transform(testMLDF)
        dt_rmse = dt_evaluator.evaluate(dt_predictions)
        # print("Decision Tree Default Parameters: ")

        # dt_params = dt_model.stages[0].params
        # for i in range(0,len(dt_params)):
        #     print(dt_params[i])
        #     print(type(dt_params[i]))

        treeModel = dt_model.stages[0]

        dt_paramMap = treeModel.extractParamMap()

        for key in dt_paramMap.keys():
            # print(key.name, dt_paramMap[key])

            if key.name in ['minInstancesPerNode']:
                minInstancesPerNode = dt_paramMap[key]
            if key.name in ['maxDepth']:
                maxDepth = dt_paramMap[key]
            if key.name in ['maxBins']:
                maxBins = dt_paramMap[key]
            if bool(params)==False:
                if key.name in ['minInstancesPerNode', 'maxDepth', 'maxBins']:
                    print(key.name, dt_paramMap[key])
                    print(key.doc)

        print("Decision Tree Root Mean Squared Error (RMSE) on test data = %g" % dt_rmse)
        selected = dt_predictions.select("prediction","atmosphereCO2","features")

        # Print presiction row by row
        # for row in selected.collect():
        #     print(row)

        return [minInstancesPerNode, maxDepth, maxBins, dt_rmse], dt_predictions.toPandas()

###################################################################
###################################################################

    def RunGradientBoostedTree(self, trainMLDF, testMLDF):

        self.FitGradientBoostedTree(trainMLDF, testMLDF)

        # Best parameters after running the following
        # 1	10	36	25	0.05	1	37921.1148178837
        # [minInstancesPerNode, maxDepth, maxBins, maxIter, stepSize, subsamplingRate, gbt_rmse]

        # Deafault
        gbt_maxDepth_array = [5]
        gbt_maxBins_array = [32]
        gbt_minInstancesPerNode_array = [1]
        gbt_maxIter_array = [20]
        gbt_stepSize_array = [0.1]
        gbt_subsamplingRate_array = [1.0]

        # Optimal
        # gbt_maxDepth_array = [10]
        # gbt_maxBins_array = [36]
        # gbt_minInstancesPerNode_array = [1]
        # gbt_maxIter_array = [25]
        # gbt_stepSize_array = [0.05]
        # gbt_subsamplingRate_array = [1.0]

        # Swap for these to running the hyper-parameter search again
        # gbt_maxDepth_array = [10,20,30]
        # gbt_maxBins_array = [32,34,36]
        # gbt_minInstancesPerNode_array = [1,2,3]
        # gbt_maxIter_array = [20, 25, 30]
        # gbt_stepSize_array = [0.05, 0.1, 0.2]
        # gbt_subsamplingRate_array = [0.4, 0.8, 1.0]

        gbt_results_array = [] # minInstancesPerNode, maxDepth, maxBins, rmse

        for i in range(0, len(gbt_maxDepth_array)):
            for j in range(0, len(gbt_maxBins_array)):
                for k in range(0, len(gbt_minInstancesPerNode_array)):
                    for l in range(0, len(gbt_maxIter_array)):
                        for m in range(0, len(gbt_stepSize_array)):
                            for n in range(0, len(gbt_subsamplingRate_array)):
                                gbt_maxDepth = gbt_maxDepth_array[i]
                                gbt_maxBins = gbt_maxBins_array[j]
                                gbt_minInstancesPerNode = gbt_minInstancesPerNode_array[k]
                                gbt_maxIter = gbt_maxIter_array[l]
                                gbt_stepSize = gbt_stepSize_array[m]
                                gbt_subsamplingRate = gbt_subsamplingRate_array[n]

                                gbt_params = {
                                    "maxDepth":gbt_maxDepth,
                                    "maxBins":gbt_maxBins,
                                    "minInstancesPerNode":gbt_minInstancesPerNode,
                                    "maxIter":gbt_maxIter,
                                    "stepSize":gbt_stepSize,
                                    "subsamplingRate":gbt_subsamplingRate
                                }

                                gbt_results, gbt_DF = self.FitGradientBoostedTree(trainMLDF, testMLDF, gbt_params)
                                gbt_results_array.append(gbt_results)

                                del gbt_maxDepth
                                del gbt_maxBins
                                del gbt_minInstancesPerNode
                                del gbt_maxIter
                                del gbt_stepSize
                                del gbt_subsamplingRate
                                del gbt_params

        for i in range(0,len(gbt_results_array)):
            print(gbt_results_array[i])

        return gbt_DF

###################################################################
###################################################################

    def FitGradientBoostedTree(self, trainMLDF, testMLDF, params={}):

        print("Running Gradient Boosted Tree.....")

        if bool(params):
            print("Fitting with maxDepth = " + str(params["maxDepth"]) + ", maxBins = " + str(params["maxBins"]) + ", minInstancesPerNode = " + str(params["minInstancesPerNode"]) + " ...")
            print("maxIter = " + str(params["maxIter"]) + ", stepSize = " + str(params["stepSize"]) + ", subsamplingRate = " + str(params["subsamplingRate"]) + " ...")
            gbt = GBTRegressor(featuresCol ='features', labelCol = 'atmosphereCO2', maxDepth=params["maxDepth"], maxBins = params["maxBins"], minInstancesPerNode = params["minInstancesPerNode"], maxIter=params["maxIter"], stepSize=params["stepSize"], subsamplingRate=params["subsamplingRate"])
        else:
            print("Fitting with default parameters...")
            gbt = GBTRegressor(featuresCol = 'features', labelCol = 'atmosphereCO2')

        gbt_pipeline = Pipeline(stages=[gbt])
        gbt_evaluator = RegressionEvaluator(labelCol="atmosphereCO2", predictionCol="prediction", metricName="rmse")

        gbt_model = gbt_pipeline.fit(trainMLDF)
        gbt_predictions = gbt_model.transform(testMLDF)
        # gbt_predictions.select('prediction', 'atmosphereCO2', 'features').show(5)
        if bool(params)==False:
            print("Gradient Boosted Tree Default Parameters: ")

        gbt_treeModel = gbt_model.stages[0]

        gbt_paramMap = gbt_treeModel.extractParamMap()
        for key in gbt_paramMap.keys():

            # print(key.name, dt_paramMap[key])
            # if bool(params):
            if key.name in ['minInstancesPerNode']:
                minInstancesPerNode = gbt_paramMap[key]
            if key.name in ['maxDepth']:
                maxDepth = gbt_paramMap[key]
            if key.name in ['maxBins']:
                maxBins = gbt_paramMap[key]
            if key.name in ['maxIter']:
                maxIter = gbt_paramMap[key]
            if key.name in ['stepSize']:
                stepSize = gbt_paramMap[key]
            if key.name in ['subsamplingRate']:
                subsamplingRate = gbt_paramMap[key]

            if bool(params)==False:
                if key.name in ['minInstancesPerNode', 'maxDepth', 'stepSize', 'maxIter', 'maxBins', 'subsamplingRate']:
                    print(key.name, gbt_paramMap[key])
                # if key.name in ['subsamplingRate']:
                    print(key.doc)


        gbt_rmse = gbt_evaluator.evaluate(gbt_predictions)
        print("Gradient Boosted Trees Root Mean Squared Error (RMSE) on test data = %g" % gbt_rmse)

        return [minInstancesPerNode, maxDepth, maxBins, maxIter, stepSize, subsamplingRate, gbt_rmse], gbt_predictions.toPandas()

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