import pandas as pd
import numpy as np
import os
import math
import random
from scipy.stats import binom
import MLCode as dg
import VariableAnalysis as si
#change the directory
##Enter the business
print("Enter Number of Business")
N = int(input())
##Variables for Normal Company'
NormalCompanyRatio = math.ceil(0.95 * N)
##Variables for Shell Company
ShellCompanyRatio = math.ceil(0.05 * N)
##Global Variables for Compnay Data
##Low,Meduium and the higj
rsize = [0.3,0.4,0.3]  #Revenue Size
##Small Medum anf the high
bsize = [0.3,0.4,0.3]  #Business Size
##1,2,3	
TQwner = [0.5,0.4,0.1] #Total Number of Qwners
##yes or no
QHistory = [0.01,0.99] ###History of the qwner
##offshore or india
LocCompany = [0.01,0.99]
##Check If the Particular location is tax haven
##Yes or no
TaxHavenLoc = [0.01,0.99]
##Number of Account Holded by the company
##1,2,3
NumberAccount = [0.5,0.4,0.1]


###Transactions Paramters are Different for both the Companys

#############################################################################################################
#############Enter Transactions Paramter for the Normal Company##########################################

#Local,Foreign and tax haven
FCurrencyN = [0.70,0.29,0.01]
#cash,cheque,transfer
##Honest company has more cash transaction
ModeTransIncomingN = [0.60,0.30,0.10]
#cash,cheque,transfer
ModeTransRetailN = [0.38,0.31,0.31]
##cash,cheque,transfer
ModeTransOutgoingN = [0.6,0.3,0.10]
##Online and offline probability
OnlineOfflineProbN = [0.6,0.4]


dg.GenerateNormalDiscrete(FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN)

#############################################################################################################
##############Enter Transactions Paramter for the Shell Company#########################################
##Local,Foreign and tax haven
FCurrencyS = [0.68,0.30,0.02]
#cash,cheque,transfer
##Honest company has more cash transaction
ModeTransIncomingS = [0.58,0.31,0.11]
#cash,cheque,transfer
ModeTransRetailS = [0.38,0.31,0.31]
##cash,cheque,transfer
ModeTransOutgoingS = [0.58,0.31,0.11]
##Online and offline probability
OnlineOfflineProbS = [0.62,0.38]
##Mthod to create list of discrete list
dg.GenerateShellDiscrete(FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS)


##Normal Variables
AccountIDNormal = 100001
SavingCurrentNormal = [0.01,0.99]
CompanyFilenameN = "CompanyData_N.csv"
##Transaction file name
TransFilenameN = "TransactionData_N.csv"
##Generate Header for the normal company
dg.GenerateHeader("Normal")
##Genrate company Data
dg.GenerateCompanyData("Normal",NormalCompanyRatio,AccountIDNormal,rsize,bsize,TQwner,QHistory,LocCompany,TaxHavenLoc,NumberAccount,SavingCurrentNormal)
##Generate Summarized file
si.SummarizedColumns(NormalCompanyRatio,"Normal",CompanyFilenameN,TransFilenameN)

##Shell Variables
AccountIDShell = 100951
##Saving and current account
SavingCurrentShell = [0.10,0.90]
##Company file name
CompanyFilenameS = "CompanyData_S.csv"
##Transaction file name
TransFilenameS = "TransactionData_S.csv"
##Genrate header for the shell company
dg.GenerateHeader("Shell")
##Generate company data for the shell company
dg.GenerateCompanyData("Shell",ShellCompanyRatio,AccountIDShell,rsize,bsize,TQwner,QHistory,LocCompany,TaxHavenLoc,NumberAccount,SavingCurrentShell)
##Generate Summarized file
si.SummarizedColumns(ShellCompanyRatio,"Shell",CompanyFilenameS,TransFilenameS)
	
NormalCompanyFilename = "CompanyData_N.csv"
NormalTransFilename = "TransactionData_N.csv"
NormalSummarizedFileName = "SummarizedData_N.csv"
ShellCompanyFilename  = "CompanyData_S.csv"
ShellTransFilename = "TransactionData_S.csv"
ShellSummarizedFileName = "SummarizedData_S.csv"
##Sumamrized Mean Table 
si.SummarizedMeanTable()
##Python Notebook with Classifier and the analysis




