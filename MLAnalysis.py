import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import time
from sklearn.model_selection import train_test_split
# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
from sklearn.metrics import roc_auc_score
import os
from sklearn import tree
import graphviz
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score,f1_score
from sklearn.metrics import precision_recall_curve,fbeta_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
os.chdir('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/FinalAnalysis1')
os.environ["PATH"] += os.pathsep + r'C:\Users\kbaja4\AppData\Local\Programs\Python\Python36\graphviz-2.38\release\bin'
from sklearn.ensemble import GradientBoostingClassifier
##PYOd Variable
from pyod.models.knn import KNN
import warnings
warnings.filterwarnings('ignore')
def Plots():
	##Check Proportion of shell and normal Company
	print('Honest Companies', round(SummarizedData['Flag'].value_counts()[0]/len(SummarizedData) * 100,2), '% of the dataset')
	print('Shell Companies', round(SummarizedData['Flag'].value_counts()[1]/len(SummarizedData) * 100,2), '% of the dataset')
	colors = ["g", "r"]
	##Plot graph
	sns.countplot('Flag', data=SummarizedData, palette=colors)
	plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/Proportion.png')
	plt.show()
	##(a) First Variable -- Foreign Transactions
	fig, ax = plt.subplots(1, 1, figsize=(18,4))
	ForeignNormal = SummarizedNormal['Foreign'].values
	ForeignShell = SummarizedShell['Foreign'].values
	sns.distplot(ForeignShell,color='r',label = "Shell")
	sns.distplot(ForeignNormal,color = 'g',label = "Normal")
	plt.title('Number of Foreign Transactions By Normal and Shell Company',fontsize = 20)
	plt.xlabel('proportion of foreign transaction/Incoming Transaction',fontsize = 20)
	plt.ylabel('Probability',fontsize = 20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/Foreign.png')
	plt.show()
	##(b) Second Variable -- Local Transactions
	fig, ax = plt.subplots(1, 1, figsize=(18,4))
	LocalNormal = SummarizedNormal['Local'].values
	LocalShell = SummarizedShell['Local'].values
	sns.distplot(LocalShell,color='r',label = "Shell")
	sns.distplot(LocalNormal,color = 'g',label = "Normal")
	plt.title('Number of Local Transactions By Normal and Shell Company',fontsize = 20)
	plt.xlabel('proportion of Local transaction/Incoming Transaction',fontsize = 20)
	plt.ylabel('Probability',fontsize = 20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/Local.png')
	plt.show()
	##(c) Tax Haven Transactions
	fig, ax = plt.subplots(1, 1, figsize=(18,4))
	TaxHavenNormal = SummarizedNormal['TaxHaven'].values
	TaxHavenShell = SummarizedShell['TaxHaven'].values
	sns.distplot(TaxHavenShell,color='r',label = "Shell")
	sns.distplot(TaxHavenNormal,color = 'g',label = "Normal")
	plt.title('Number of Tax Haven  Transactions By Normal and Shell Company',fontsize = 20)
	plt.xlabel('proportion of Tax Haven transaction/Incoming Transaction',fontsize = 20)
	plt.ylabel('Probability',fontsize = 20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/TaxHaven.png')
	plt.show()
	##(d) Client to Incoming Transaction Ratio
	fig, ax = plt.subplots(1, 1, figsize=(18,4))
	ClientToIncomingRatioNormal = SummarizedNormal['ClientToIncomingRatio'].values
	ClientToIncomingRatioShell = SummarizedShell['ClientToIncomingRatio'].values
	sns.distplot(ClientToIncomingRatioShell,color='r',label = "Shell")
	sns.distplot(ClientToIncomingRatioNormal,color = 'g',label = "Normal")
	plt.title('Distribution of the ClientToIncomingRatioShell  Transactions')
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/ClientToIncomingTransaction.png')
	plt.show()
	##(e,f) Proportion of Small and Large Amount of the Transactions
	plt.figure(figsize=(20,10))
	sns.distplot(IncomingN.TransactionAmount, bins=20,label = "Normal",color = "lightgreen")
	sns.distplot(IncomingS.TransactionAmount, bins=20,label = "Shell",color = "orangered")
	plt.xlabel('Amount Deposited by the Customer',fontsize=20)
	plt.ylabel('No of times',fontsize=20)
	plt.legend()
	plt.title('Show the Distribution of the Small Amount of the Incoming Transactions for Normal/Shell Company',fontsize=20)
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/SmallAndLargeAmount.png')
	plt.show()
	##(g) Proportion of Saving Account
	print('Honest Companies Saving Account', round(SummarizedNormal['TypeAccount'].value_counts()[0]/len(SummarizedNormal) * 100,2), '% of the dataset')
	print('Shell Companies saving Account', round(SummarizedShell['TypeAccount'].value_counts()[0]/len(SummarizedShell) * 100,2), '% of the dataset')
	##(h) Proportion of Current Account
	print('Honest Companies Saving Account', round(SummarizedNormal['TypeAccount'].value_counts()[1]/len(SummarizedNormal) * 100,2), '% of the dataset')
	print('Shell Companies saving Account', round(SummarizedShell['TypeAccount'].value_counts()[1]/len(SummarizedShell) * 100,2), '% of the dataset')
	##(i) Proportion of Cash Transactions
	fig, ax = plt.subplots(1, 1, figsize=(18,4))
	IncomingCashTransactiontNormal = SummarizedNormal['IncomingCashTransaction'].values
	IncomingCashTransactionShell = SummarizedShell['IncomingCashTransaction'].values
	sns.distplot(IncomingCashTransactionShell,color='r',label = "Shell")
	sns.distplot(IncomingCashTransactiontNormal,color = 'g',label = "Normal")
	plt.title('Cash Deposited by the Customer in Normal/Shell Company',fontsize = 20)
	plt.xlabel('Percentage Of Amount Deposited',fontsize = 20)
	plt.ylabel('Probabilities',fontsize = 20)
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/CashTransactions.png')
	plt.legend()
	plt.show()
	##(j)Proportion of Online Transaction
	SummarizedShell['IncomingOnlineTransaction'] = pd.to_numeric(SummarizedShell['IncomingOnlineTransaction'], errors='coerce')
	fig, ax = plt.subplots(1, 1, figsize=(18,4))
	IncomingOnlineTransactionNormal = SummarizedNormal['IncomingOnlineTransaction'].values
	IncomingOnlineTransactionShell = SummarizedShell['IncomingOnlineTransaction'].values
	sns.distplot(IncomingOnlineTransactionShell,color='r',label = "Shell")
	sns.distplot(IncomingOnlineTransactionNormal,color = 'g',label = "Normal")
	plt.title('Online Transfer of Money By Normal And Shell Company',fontsize = 20)
	plt.xlabel('Percentage of Online Transfer',fontsize=20)
	plt.ylabel('Probabilities',fontsize=20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/OnlineTransactions.png')
	plt.legend()
	plt.show()
	##(k)Proportion of Annual Expenditure
	fig, ax = plt.subplots(1, 1, figsize=(18,4))
	ExpenditureAmountNormal = CompanyDataN['ExpenditureAmount'].values
	ExpenditureAmountShell = CompanyDataS['ExpenditureAmount'].values
	sns.distplot(ExpenditureAmountShell,color='r',label = "Shell")
	sns.distplot(ExpenditureAmountNormal,color = 'g',label = "Normal")
	plt.title('ExpenditureAmount By Normal And Shell Company',fontsize = 20)
	plt.xlabel('Percentage of ExpenditureAmount',fontsize=20)
	plt.ylabel('Probabilities',fontsize=20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/ExpenditureAmount.png')
	plt.legend()
	plt.show()
	##(L & M) Small and Large Amount of Outgoing Transaction
	##(e,f) Proportion of Small and Large Amount of the OutgoingTransactions
	plt.figure(figsize=(20,10))
	sns.distplot(OutgoingN.TransactionAmount, bins=20,label = "Normal",color = "lightgreen")
	sns.distplot(Outgoing.TransactionAmount, bins=20,label = "Shell",color = "orangered")
	plt.xlabel('Amount Withdrawal by the Customer',fontsize=20)
	plt.ylabel('No of times',fontsize=20)
	plt.legend()
	plt.title('Small&Large Amount of Debit Transactions by Normal/Shell Company for Year',fontsize=20)
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/SmallAndLargeAmountOutgoing.png')
	plt.show()
	##(0) Average Transaction Per Client 
	fig, ax = plt.subplots(1, 1, figsize=(18,4))
	AvgTransPerClientNormal = SummarizedNormal['AvgTransPerClient'].values
	AvgTransPerClientShell = SummarizedShell['AvgTransPerClient'].values
	sns.distplot(AvgTransPerClientShell,color='r',label = "Shell")
	sns.distplot(AvgTransPerClientNormal,color = 'g',label = "Normal")
	plt.title('AvgTransPerClient By Normal And Shell Company',fontsize = 20)
	plt.xlabel('Percentage of AvgTransPerClientNormal',fontsize=20)
	plt.ylabel('Probabilities',fontsize=20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/AvgTransPerClient.png')
	plt.show()
	##(P)Incoming Transaction Percentage
	fig, ax = plt.subplots(1, 1, figsize=(18,4))
	AvgTransPerClientNormal = SummarizedNormal['IncomingTransPercentage'].values
	AvgTransPerClientShell = SummarizedShell['IncomingTransPercentage'].values
	sns.distplot(AvgTransPerClientShell,color='r',label = "Shell")
	sns.distplot(AvgTransPerClientNormal,color = 'g',label = "Normal")
	plt.title('IncomingTransPercentage By Normal And Shell Company',fontsize = 20)
	plt.xlabel('Percentage of IncomingTransPercentage',fontsize=20)
	plt.ylabel('Probabilities',fontsize=20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/AvgTransPerClient.png')
	plt.show()
	##Outgoing Transaction percentage
	fig, ax = plt.subplots(1, 1, figsize=(18,10))
	OutgoingTransPercentageNormal = SummarizedNormal['OutgoingTransPercentage'].values
	OutgoingTransPercentageShell = SummarizedShell['OutgoingTransPercentage'].values
	sns.distplot(OutgoingTransPercentageShell,color='r',label = "Shell")
	sns.distplot(OutgoingTransPercentageNormal,color = 'g',label = "Normal")
	plt.title('OutgoingTransPercentage By Normal And Shell Company',fontsize = 20)
	plt.xlabel('Percentage of OutgoingTransPercentage',fontsize=20)
	plt.ylabel('Probabilities',fontsize=20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/OutgoingTransPercentage.png')
	plt.legend()
	plt.show()
	##(R) Average Incoming Transaction Amount(Montly)
	fig, ax = plt.subplots(1, 1, figsize=(18,10))
	AvgIncomingAmountNormal = SummarizedNormal['AvgIncomingAmount'].values
	AvgIncomingAmountShell = SummarizedShell['AvgIncomingAmount'].values
	sns.distplot(AvgIncomingAmountShell,color='r',label = "Shell")
	sns.distplot(AvgIncomingAmountNormal,color = 'g',label = "Normal")
	plt.title('AvgIncomingAmount ByNormal And Shell Company',fontsize = 20)
	plt.xlabel('Percentage of AvgIncomingAmount',fontsize=20)
	plt.ylabel('Probabilities',fontsize=20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/AvgIncomingAmount.png')
	plt.show()
	##(S) Average Outgoing Transaction Amount(Montly)
	fig, ax = plt.subplots(1, 1, figsize=(18,10))
	AvgOutgoingAmountNormal = SummarizedNormal['AvgOutgoingAmount'].values
	AvgOutgoingAmountShell = SummarizedShell['AvgOutgoingAmount'].values
	sns.distplot(AvgOutgoingAmountShell,color='r',label = "Shell")
	sns.distplot(AvgOutgoingAmountNormal,color = 'g',label = "Normal")
	plt.title('AvgOutgoingAmount ByNormal And Shell Company',fontsize = 20)
	plt.xlabel('Percentage of AvgOutgoingAmount',fontsize=20)
	plt.ylabel('Probabilities',fontsize=20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/AvgOutgoingAmount.png')
	plt.show()
	##(U) Average Number of Incoming Transaction(Montly)
	fig, ax = plt.subplots(1, 1, figsize=(18,10))
	AvgIncomingCountNormal = SummarizedNormal['AvgIncomingCount'].values
	AvgIncomingCountShell = SummarizedShell['AvgIncomingCount'].values
	sns.distplot(AvgIncomingCountShell,color='r',label = "Shell")
	sns.distplot(AvgIncomingCountNormal,color = 'g',label = "Normal")
	plt.title('AvgIncomingCount ByNormal And Shell Company',fontsize = 20)
	plt.xlabel('Percentage of AvgIncomingCount',fontsize=20)
	plt.ylabel('Probabilities',fontsize=20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/AvgIncomingCount.png')
	plt.show()
	##(T) Average Number of Outgoing Transaction(Montly)
	fig, ax = plt.subplots(1, 1, figsize=(18,10))
	AvgOutgoingCountNormal = SummarizedNormal['AvgOutgoingCount'].values
	AvgOutgoingCountShell = SummarizedShell['AvgOutgoingCount'].values
	sns.distplot(AvgOutgoingCountShell,color='r',label = "Shell")
	sns.distplot(AvgOutgoingCountNormal,color = 'g',label = "Normal")
	plt.title('AvgOutgoingCount ByNormal And Shell Company',fontsize = 20)
	plt.xlabel('Percentage of  AvgOutgoingCount',fontsize=20)
	plt.ylabel('Probabilities',fontsize=20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/AvgOutgoingCount.png')
	plt.show()
	##(U)Percentage of Expenditure
	fig, ax = plt.subplots(1, 1, figsize=(18,10))
	PercentageExpenditureNormal = SummarizedNormal['PercentageExpenditure'].values
	PercentageExpenditureShell = SummarizedShell['PercentageExpenditure'].values
	sns.distplot(PercentageExpenditureShell,color='r',label = "Shell")
	sns.distplot(PercentageExpenditureNormal,color = 'g',label = "Normal")
	plt.title('PercentageExpenditure ByNormal And Shell Company',fontsize = 20)
	plt.xlabel('Percentage of  PercentageExpenditure',fontsize=20)
	plt.ylabel('Probabilities',fontsize=20)
	plt.legend()
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/PercentageExpenditure.png')
	plt.show()
	
def PrintMetrics(ModelName,Y_train,TrainPred,Y_test,best_predictions):
	print("---------------------------------Accuracy for training Data-------------------------------------------------")
	print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(Y_train,TrainPred)))
	print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(Y_train,TrainPred,beta = 1.5)))
	print('precision_score',precision_score(Y_train,TrainPred))
	print('recall_score',recall_score(Y_train,TrainPred))
	print("--------------------------------Accuracy for Testing Data----------------------------------------------------")
	print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(Y_test,best_predictions)))
	print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(Y_test,best_predictions,beta = 1.5)))
	print('precision_score',precision_score(Y_test,best_predictions))
	print('recall_score',recall_score(Y_test,best_predictions))
	cm = confusion_matrix(Y_test,best_predictions,labels = [1,0])
	print(cm)
	fpr, tpr, thresholds = roc_curve(Y_test,best_predictions)
	fig, ax = plt.subplots(1, figsize=(12, 6))
	plt.plot(fpr, tpr, color='darkorange', label='Model Performace')
	plt.plot([0, 1], [0, 1], color='gray', label='Random Performace')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Honest/Shell Analysis ROC Curve for' + ModelName)
	plt.legend(loc="lower right")
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/' + ModelName + 'AUC Curve.png')
	print('Auc Score is : ', roc_auc_score(Y_test,best_predictions))
	
def Correlation():
	##Pearson Correlation
	colormap = plt.cm.viridis
	plt.figure(figsize=(20,20))
	plt.title('Pearson Correlation of Features', y=1.05, size=15)
	sns.heatmap(SummarizedData.astype(float).corr(),linewidths=0,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/CorrelationMatrix.png')
	plt.show()
	CorrelationTartgetValue = SummarizedData[SummarizedData.columns[1:]].corr()['Flag'][:].sort_values(ascending = False)
	CorrelationTartgetValue
	plt.figure(figsize=(20,15))
	#CorrelationTartgetValue = CorrelationTartgetValue.drop(['Flag'])
	CorrelationTartgetValue.plot(x ='Country', y='GDP_Per_Capita', kind = 'bar')
	plt.title('Corrrelation with the Target Variable',fontsize = 24)
	plt.xlabel('Features',fontsize = 24)
	plt.ylabel('Correlated Value',fontsize = 24)
	plt.savefig('C:/Users/kbaja4/AppData/Local/Programs/Python/Python36/Scripts/Output/CorrelationWithFlagVariable.png')
	plt.show()
	
def plot_feature_importance(fi):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(24,8))
    ax1.plot(np.arange(0, len(fi.index)), fi['importance'])
    label_nrs = np.arange(0, len(fi.index), 5 )
    ax1.set_xticks(label_nrs)
    ax1.set_xticklabels(fi['feature'][label_nrs], rotation=90)    
    num_bar = min(len(fi.index), 30)
    ax2.barh(np.arange(0, num_bar), fi['importance'][:num_bar], align='center', alpha=0.5)
    ax2.set_yticks(np.arange(0, num_bar))
    ax2.set_yticklabels(fi['feature'][:num_bar])
	

	
def SetVariableAnalysis():
	##AdaBoostClassifiers
	print("-----------------AdaBoostClassifier---------------------------------------")
	from sklearn.ensemble import AdaBoostClassifier
	param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
				  "base_estimator__splitter" :   ["best", "random"],
				  "n_estimators": [1, 2,3,4,5,6,7,8],"learning_rate":[0.5,1,1.5,2,2.5,3.5]
				 }
	DTC = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
						   max_features=None, max_leaf_nodes=None,
						   min_impurity_decrease=0.0, min_impurity_split=None,
						   min_samples_leaf=3, min_samples_split=2,
						   min_weight_fraction_leaf=0.0, presort=False,
						   random_state=None, splitter='best')

	ABC = AdaBoostClassifier(base_estimator = DTC)

	# run grid search
	grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')
	grid_fit = grid_search_ABC.fit(X_train,Y_train)
	best_clf = grid_fit.best_estimator_
	print("Best Esimator",grid_fit.best_estimator_)
	best_predictions = best_clf.predict(X_test)

	TrainPred = best_clf.predict(X_train)
	#Print Metrics for each classifier
	PrintMetrics("AdaBoost",Y_train,TrainPred,Y_test,best_predictions)
	
	
	
def PCAModels():
	##AdaBoostClassifiers
	print("-----------------AdaBoostClassifier---------------------------------------")
	from sklearn.ensemble import AdaBoostClassifier
	param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
				  "base_estimator__splitter" :   ["best", "random"],
				  "n_estimators": [1, 2,3,4,5,6,7,8],"learning_rate":[0.5,1,1.5,2,2.5,3.5]
				 }
	DTC = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
						   max_features=None, max_leaf_nodes=None,
						   min_impurity_decrease=0.0, min_impurity_split=None,
						   min_samples_leaf=3, min_samples_split=2,
						   min_weight_fraction_leaf=0.0, presort=False,
						   random_state=None, splitter='best')

	ABC = AdaBoostClassifier(base_estimator = DTC)

	# run grid search
	grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')
	grid_fit = grid_search_ABC.fit(X_train,Y_train)
	best_clf = grid_fit.best_estimator_
	print("Best Esimator",grid_fit.best_estimator_)
	best_predictions = best_clf.predict(X_test)

	TrainPred = best_clf.predict(X_train)
	#Print Metrics for each classifier
	PrintMetrics("AdaBoost",Y_train,TrainPred,Y_test,best_predictions)
	
def KNNAlgo(TrainX,TestX,TrainY,TestY):
	##Copy Variabel for Accuracy Analysis
	CopyTrainY = TrainY.copy()
	CopyTestY = TestY.copy()
	##Applying KNN algorith,
	clf = KNN(n_neighbors=20)
	clf.fit(TrainX)
	##Predicting Label for training Dataset
	y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
	##Outlier Scores for training Data Points
	y_train_scores = clf.decision_scores_  # raw outlier scores
	##Predicting Label for Test Dataset
	y_test_pred = clf.predict(TestX)  # outlier labels (0 or 1)
	##Outlier scores for test dataset
	y_test_scores = clf.decision_function(TestX)  # outlier scores
	##Plot Outlier scoring Points for training Dataset
	sns.distplot(y_train_scores)
	plt.title('Distance from the Kth Nearest Neighbour')
	plt.savefig('KthDistanceTrainingSet.png')
	##Plot Outlier scoring Points for test Dataset
	sns.distplot(y_test_scores)
	plt.title('Distance from the Kth Nearest Neighbour')
	plt.savefig('KthDistanceTestSet.png')
	##Creating A dataframe for Train Dataset consisting of the (Outlier score + Label + Xpoints)
	TrainScores = list(y_train_scores)##Outlier scores of training Dataset
	XPoints = np.arange(1,len(TrainScores)+1)##X axis
	CombineTrainFile = TrainY#Labels for training Dataset
	CombineTrainFile['XPoints'] = XPoints##0,1,2 -------------length of training set
	CombineTrainFile['TrainScores'] = TrainScores##Oulier scores
	##Plot Scatter Plot for the Local Outlier Scores
	colors = ['green','red']
	fig = plt.figure(figsize=(8,8))
	plt.scatter(CombineTrainFile['XPoints'],CombineTrainFile['TrainScores'], c=CombineTrainFile['Flag'],cmap=matplotlib.colors.ListedColormap(colors))
	plt.title('Outliers in Traning Set for Normal and Shell Company')
	plt.legend()
	##Creating A dataframe for est Dataset consisting of the (Outlier score + Label + Xpoints)
	TestScores = list(y_test_scores)##Outlier scores of training Dataset
	XPoints = np.arange(1,len(TestScores)+1)##X axis
	CombineTestFile = TestY#Labels for training Dataset
	CombineTestFile['XPoints'] = XPoints##0,1,2 -------------length of training set
	CombineTestFile['TestScores'] = TestScores##Oulier scores
	##Plot Scatter Plot for the Local Outlier Scores
	colors = ['green','red']
	fig = plt.figure(figsize=(8,8))
	plt.scatter(CombineTestFile['XPoints'],CombineTestFile['TestScores'], c=CombineTestFile['Flag'],cmap=matplotlib.colors.ListedColormap(colors))
	plt.title('Outliers in Test Set for Normal and Shell Company')
	plt.legend()
	###Check the Accuracy of the KNN model
	print("---------------------------------Accuracy for training Data-------------------------------------------------")
	print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(CopyTrainY,y_train_pred)))
	print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(CopyTrainY,y_train_pred,beta = 1.2)))
	print('precision_score',precision_score(CopyTrainY,y_train_pred))
	print('recall_score',recall_score(CopyTrainY,y_train_pred))
	print("--------------------------------Accuracy for Testing Data----------------------------------------------------")
	print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(CopyTestY,y_test_pred)))
	print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(CopyTestY,y_test_pred,beta = 1.2)))
	print('precision_score',precision_score(CopyTestY,y_test_pred))
	print('recall_score',recall_score(CopyTestY,y_test_pred))
	cm = confusion_matrix(CopyTestY,y_test_pred,labels = [1,0])
	print("Confusion matrix for test dataset")
	print(cm)
	fpr, tpr, thresholds = roc_curve(CopyTestY,y_test_pred)
	fig, ax = plt.subplots(1, figsize=(12, 6))
	plt.plot(fpr, tpr, color='darkorange', label='Model Performace')
	plt.plot([0, 1], [0, 1], color='gray', label='Random Performace')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Honest/Shell Analysis ROC Curve for KNN Outlier Detection Algorithm for test Dataset')
	plt.legend(loc="lower right")
	print('Auc Score is : ', roc_auc_score(CopyTestY,y_test_pred))

def Models():
	##Applying Decision Tree
	print("-------------------Decision Tree Classifier----------------------------")
	clf = tree.DecisionTreeClassifier(criterion='entropy')

	parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)},{'min_samples_split': (0.1,0.2,0.3,0.4,0.5)},{'min_samples_leaf': (1,2,3,4,5,6,7)},{'min_weight_fraction_leaf': (0.0,0.1,0.2),"max_features":[10,11,12,13]}
	scorer = make_scorer(fbeta_score,beta=1.5)

	grid_obj = GridSearchCV(clf, param_grid = parameters, scoring=scorer)
	grid_fit = grid_obj.fit(X_train,Y_train)

	best_clf = grid_fit.best_estimator_
	print("Best Esimator",grid_fit.best_estimator_)
	best_predictions = best_clf.predict(X_test)
	##Print Tree Structure
	dot_data = tree.export_graphviz(best_clf, out_file=None, max_depth=10, feature_names=list(X_train.columns.values), filled=True, rounded=True)
	valgTre = graphviz.Source(dot_data) 
	png_bytes = valgTre.pipe(format='png')
	with open('DecisionTree_Classifier.png','wb') as f:
		f.write(png_bytes)
	#from IPython.display import Image
	#Image(png_bytes)
	valgTre
	TrainPred = best_clf.predict(X_train)
	#Print Metrics for each classifier
	PrintMetrics("Decision Tree",Y_train,TrainPred,Y_test,best_predictions)	
	##Ado Boosr
	##AdaBoostClassifiers
	print("-----------------AdaBoostClassifier---------------------------------------")
	from sklearn.ensemble import AdaBoostClassifier
	param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
				  "base_estimator__splitter" :   ["best", "random"],
				  "n_estimators": [1, 2,3,4,5,6,7,8],"learning_rate":[0.5,1,1.5,2,2.5,3.5]
				 }
	DTC = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
						   max_features=None, max_leaf_nodes=None,
						   min_impurity_decrease=0.0, min_impurity_split=None,
						   min_samples_leaf=3, min_samples_split=2,
						   min_weight_fraction_leaf=0.0, presort=False,
						   random_state=None, splitter='best')

	ABC = AdaBoostClassifier(base_estimator = DTC)

	# run grid search
	grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')
	grid_fit = grid_search_ABC.fit(X_train,Y_train)
	best_clf = grid_fit.best_estimator_
	print("Best Esimator",grid_fit.best_estimator_)
	best_predictions = best_clf.predict(X_test)

	TrainPred = best_clf.predict(X_train)
	#Print Metrics for each classifier
	PrintMetrics("AdaBoost",Y_train,TrainPred,Y_test,best_predictions)	
	##XgBoost
	print("---------------------------xgboost Classifier----------------------------------")
	import xgboost as xgb
	model2 = xgb.XGBClassifier(n_estimators=5, max_depth=8, learning_rate=1.5, subsample=0.5)
	train_model1 = model2.fit(X_train, Y_train)
	pred1 = train_model1.predict(X_test)
	train_pred = train_model1.predict(X_train)
	best_clf = grid_fit.best_estimator_
	print("Best Esimator",grid_fit.best_estimator_)
	best_predictions = best_clf.predict(X_test)
	TrainPred = best_clf.predict(X_train)
	#Print Metrics for each classifier
	PrintMetrics("xgboost",Y_train,TrainPred,Y_test,best_predictions)
	print("XGBoost plot importance")
	xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric="auc")

	#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

	xgb_model.fit(X_train, Y_train, early_stopping_rounds=10,eval_set=[(X_test, Y_test)], verbose=False)

	xgb.plot_importance(xgb_model)

	# plot the output tree via matplotlib, specifying the ordinal number of the target tree
	# xgb.plot_tree(xgb_model, num_trees=xgb_model.best_iteration)

	# converts the target tree to a graphviz instance
	xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration)
	print("------------------------Random Forest Classifier-----------------------------")
	##Random Forest Classifier
	param_grid = { 
    'n_estimators': [5,10,20],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,10,20],
    'criterion' :['gini', 'entropy']
	}
	clf=RandomForestClassifier(random_state=42)	
	scorer = make_scorer(fbeta_score,beta=1.5)
	grid_obj = GridSearchCV(clf, param_grid = param_grid, scoring=scorer,cv = 5)
	grid_fit = grid_obj.fit(X_train,Y_train)
	best_clf = grid_fit.best_estimator_
	print("Best Esimator",grid_fit.best_estimator_)
	best_predictions = best_clf.predict(X_test)
	TrainPred = best_clf.predict(X_train)
	#Print Metrics for each classifier
	PrintMetrics("RandomForestClassifier",Y_train,TrainPred,Y_test,best_predictions)	
	m = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=8, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=20,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
	m.fit(X_train, Y_train)
	fi = pd.DataFrame({'feature': X_train.columns, 'importance': m.feature_importances_}).sort_values(by='importance', ascending=False)
	fi = fi.reset_index()
	fi
	print("Feature Importance of Random Forest")
	plot_feature_importance(fi)
	print("------------------------------GradientBoostingClassifier-------------------------------------")
	##GradientBoostingClassifier
	parameters = {
    'n_estimators': [5,10], 'learning_rate': [0.5, 1.0]
    }
	clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

	grid_fit = clf.fit(X_train,Y_train.values.ravel())

	best_clf = grid_fit.best_estimator_
	print("Best Esimator",grid_fit.best_estimator_)
	best_predictions = best_clf.predict(X_test)

	TrainPred = best_clf.predict(X_train)
	#Print Metrics for each classifier
	PrintMetrics("GradientBoostingClassifier",Y_train,TrainPred,Y_test,best_predictions)	

	
if __name__ == '__main__':
	#Read Summarized Files	
	SummarizedNormal = pd.read_csv('SummarizedData_N.csv')
	SummarizedShell =  pd.read_csv('SummarizedData_S.csv')
	SummarizedData = SummarizedNormal.append(SummarizedShell)
	# creating a dict file  
	Company = {'S': 1,'N': 0} 
	SummarizedData.Flag = [Company[item] for item in SummarizedData.Flag] 
	##Checking Null values
	SummarizedData.isnull().sum().max()
	SummarizedNormal.fillna( method ='ffill', inplace = True)
	SummarizedShell.fillna( method ='ffill', inplace = True) 
	SummarizedData.fillna( method ='ffill', inplace = True) 
	##Checking Null values
	SummarizedData.isnull().sum().max()
	##Drop Account Number
	SummarizedData = SummarizedData.drop(['AccountNumber'],axis=1)
	##Read Multiple files and combine files
	CompanyDataN = pd.read_csv('CompanyData_N.csv')
	CompanyDataS = pd.read_csv('CompanyData_S.csv')
	CompanyData = CompanyDataN.append(CompanyDataS)
	TransactionDataN = pd.read_csv('TransactionData_N.csv')
	TransactionDataS = pd.read_csv('TransactionData_S.csv')
	TransactionData = TransactionDataN.append(TransactionDataS)
	IncomingN = TransactionDataN[TransactionDataN.TransactionCategory == "Credit"]
	IncomingS = TransactionDataS[TransactionDataS.TransactionCategory == "Credit"]
	OutgoingN = TransactionDataN[TransactionDataN.TransactionCategory == "Debit"]
	Outgoing = TransactionDataS[TransactionDataS.TransactionCategory == "Debit"]
	print("Starting printing Plots")
	Plots()
	print("Print Correlation matrix")
	Correlation()	
	X = SummarizedData.iloc[:, SummarizedData.columns != 'Flag']
	Y = SummarizedData.iloc[:, SummarizedData.columns == 'Flag']
	# Using All dataset split X,Y
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)
	print("Number transactions train dataset: ", len(X_train))
	print("Number transactions test dataset: ", len(X_test))
	print("Total number of transactions: ", len(X_train)+len(X_test))
	print("Check Accuracy using All Dataset")
	Models()	
	##Remove Some Variables and then check accuracy for the different Models
	SummarizedData_copy = SummarizedData.copy()
	SummarizedData_copy = SummarizedData_copy.drop(['AnnualRevenue','AvgIncomingCount','AvgIncomingAmount','TypeAccount','Foreign','AnnualProfit'], axis = 1)
	X = SummarizedData_copy.iloc[:, SummarizedData_copy.columns != 'Flag']
	Y = SummarizedData_copy.iloc[:, SummarizedData_copy.columns == 'Flag']
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)
	print("Number transactions train dataset: ", len(X_train))
	print("Number transactions test dataset: ", len(X_test))
	print("Total number of transactions: ", len(X_train)+len(X_test))
	print("Check Accuracy after removal of some  variables")
	Models()
	##Applying PCA
	print("Applying PCA-----------------------------------------")
	SummarizedData_copy1 = SummarizedData.copy()
	X = SummarizedData_copy1.iloc[:, SummarizedData_copy1.columns != 'Flag']
	Y = SummarizedData_copy1.iloc[:, SummarizedData_copy1.columns == 'Flag']
	scaler = StandardScaler()
	X1 = scaler.fit_transform(X)
	pca = PCA()
	pca.fit_transform(X1)
	pca.get_covariance()
	explained_variance=pca.explained_variance_ratio_
	with plt.style.context('dark_background'):
		plt.figure(figsize=(15,15))

		plt.bar(range(22), explained_variance, alpha=0.5, align='center',
				label='individual explained variance')
		plt.ylabel('Explained variance ratio')
		plt.xlabel('Principal components')
		plt.legend(loc='best')
		plt.tight_layout()
		plt.savefig('PCAPlot_BeforPCA.png')
	pca=PCA(n_components=13)
	X_new=pca.fit_transform(X1)
	pca.get_covariance()
	explained_variance=pca.explained_variance_ratio_
	with plt.style.context('dark_background'):
		plt.figure(figsize=(6, 4))

		plt.bar(range(13), explained_variance, alpha=0.5, align='center',
				label='individual explained variance')
		plt.ylabel('Explained variance ratio')
		plt.xlabel('Principal components')
		plt.legend(loc='best')
		plt.tight_layout()
		plt.savefig('PCAPlot_AfrerPCA.png')
	X_train, X_test, Y_train, Y_test = train_test_split(X_new,Y,test_size = 0.3, random_state = 0)
	print("Number transactions train dataset: ", len(X_train))
	print("Number transactions test dataset: ", len(X_test))
	print("Total number of transactions: ", len(X_train)+len(X_test))
	print("Check After applying PCA")
	PCAModels()
	##Check The Analysis using Set of Variables
	#(A) Check Feature Importance and try to make set of variables for the Analysis
	print("Analysis using the Combination of variables and check accuracy")
	SummarizedData_copy10 = SummarizedData.copy()
	X10 = SummarizedData_copy10.iloc[:, SummarizedData_copy10.columns != 'Flag']
	Y10 = SummarizedData_copy10.iloc[:, SummarizedData_copy10.columns == 'Flag']
	scaler = MinMaxScaler() 
	rescaledX10  = scaler.fit_transform(X10)
	X_train, X_test, Y_train, Y_test = train_test_split(rescaledX10,Y10,test_size = 0.3, random_state = 0)
	print("Number transactions train dataset: ", len(X_train))
	print("Number transactions test dataset: ", len(X_test))
	print("Total number of transactions: ", len(X_train)+len(X_test))
	m = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=8, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=20,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
	m.fit(X_train,Y_train)
	fi = pd.DataFrame({'feature': X10.columns, 'importance': m.feature_importances_}).sort_values(by='importance', ascending=False)
	fi = fi.reset_index()
	fi
	X11 = SummarizedData_copy10.iloc[:, SummarizedData_copy10.columns == 'TaxHaven']
	Y11 = SummarizedData_copy10.iloc[:, SummarizedData_copy10.columns == 'Flag']
	scaler = MinMaxScaler() 
	rescaledX11  = scaler.fit_transform(X11)
	# Whole dataset
	X_train,X_test, Y_train,Y_test = train_test_split(rescaledX11,Y11,test_size = 0.3, random_state = 0)
	SetVariableAnalysis()
	##Use of 2 variables 
	X11 = SummarizedData_copy10[['TaxHaven','ClientToIncomingRatio']]
	Y11 = SummarizedData_copy10.iloc[:, SummarizedData_copy10.columns == 'Flag']
	scaler = MinMaxScaler() 
	rescaledX11  = scaler.fit_transform(X11)
	# Whole dataset
	X_train,X_test, Y_train,Y_test = train_test_split(rescaledX11,Y11,test_size = 0.3, random_state = 0)
	SetVariableAnalysis()
	##Use of 3 variables
	X11 = SummarizedData_copy10[['TaxHaven','ClientToIncomingRatio','OutgoingTransPercentage']]
	Y11 = SummarizedData_copy10.iloc[:, SummarizedData_copy10.columns == 'Flag'] 
	scaler = MinMaxScaler() 
	rescaledX11  = scaler.fit_transform(X11)
	# Whole dataset
	X_train,X_test, Y_train,Y_test = train_test_split(rescaledX11,Y11,test_size = 0.3, random_state = 0)
	SetVariableAnalysis()
	##Use of 4 variables
	X11 = SummarizedData_copy10[['TaxHaven','ClientToIncomingRatio','OutgoingTransPercentage','IncomingTransLargeAmount']]
	Y11 = SummarizedData_copy10.iloc[:, SummarizedData_copy10.columns == 'Flag']
	scaler = MinMaxScaler() 
	rescaledX11  = scaler.fit_transform(X11)
	# Whole dataset
	X_train,X_test, Y_train,Y_test = train_test_split(rescaledX11,Y11,test_size = 0.3, random_state = 0)
	SetVariableAnalysis()
	##Use of 5 variables
	X11 = SummarizedData_copy10[['TaxHaven','ClientToIncomingRatio','OutgoingTransPercentage','IncomingTransLargeAmount','AnnualExpenditure']]
	Y11 = SummarizedData_copy10.iloc[:, SummarizedData_copy10.columns == 'Flag']
	scaler = MinMaxScaler() 
	rescaledX11  = scaler.fit_transform(X11)
	# Whole dataset
	X_train,X_test, Y_train,Y_test = train_test_split(rescaledX11,Y11,test_size = 0.30, random_state = 0)
	SetVariableAnalysis()
	##Use of 6 variables
	X11 = SummarizedData_copy10[['TaxHaven','ClientToIncomingRatio','OutgoingTransPercentage','IncomingTransLargeAmount','AnnualExpenditure','AvgOutgoingAmount']]
	Y11 = SummarizedData_copy10.iloc[:, SummarizedData_copy10.columns == 'Flag']
	scaler = MinMaxScaler() 
	rescaledX11  = scaler.fit_transform(X11)
	# Whole dataset
	X_train,X_test, Y_train,Y_test = train_test_split(rescaledX11,Y11,test_size = 0.3, random_state = 0)
	SetVariableAnalysis()
	##Use of 7 variables
	X11 = SummarizedData_copy10[['TaxHaven','ClientToIncomingRatio','OutgoingTransPercentage','IncomingTransLargeAmount','AnnualExpenditure','AvgIncomingCount']]
	Y11 = SummarizedData_copy10.iloc[:, SummarizedData_copy10.columns == 'Flag'] 
	scaler = MinMaxScaler() 
	rescaledX11  = scaler.fit_transform(X11)
	# Whole dataset
	X_train,X_test, Y_train,Y_test = train_test_split(rescaledX11,Y11,test_size = 0.3, random_state = 0)
	SetVariableAnalysis()
	##Check Analysis using Outlier Detection Algorithm
	##First Applied KNN algorithm
	SummarizedData_copy = SummarizedData.copy()
	X = SummarizedData_copy.iloc[:, SummarizedData_copy.columns != 'Flag']
	Y = SummarizedData_copy.iloc[:, SummarizedData_copy.columns == 'Flag']
	scaler = MinMaxScaler() 
	rescaledX = scaler.fit_transform(X)
	TrainX,TestX,TrainY,TestY = train_test_split(rescaledX,Y,test_size = 0.2, random_state = 0)
	KNNAlgo(TrainX,TestX,TrainY,TestY)
	

	
	
	
	
	
	

	
	
	

	
	
	

	

	
	
	
	
	
	
	
	


