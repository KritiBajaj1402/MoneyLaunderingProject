import csv
from scipy.stats import uniform
import pandas as pd
import numpy as np
import random
from scipy import stats
from scipy.stats import binom
import datetime
import calendar
import time
from scipy.stats import bernoulli
import os
#os.environ["PATH"] += os.pathsep + 'C:\Users\kbaja4\AppData\Local\Programs\Python\Python36\graphviz-2.38\release\bin'

##Count true postive in the data
CountTruePositive = 0
##Global Variables for the company Data
Business = ['Construction','Jewellery','Restaurant','Real Estate Agency','Equipment Hiring','Electronics','Manufacturing','Clothes Manufacturing','Foundry','Chemical Manufacturing','Finance','Insurance Agent']
#Business Category
BusinessCatg = ['Retail','SemiBulk','Bulk']
#Assign Business Category to the Business
BusinessSize = {"Construction":"Bulk","Jewellery":"Retail","Restaurant":"Retail","Real Estate Agency":"Bulk","Equipment Hiring":"Bulk","Electronics":"Retail","Manufacturing":"Bulk","Clothes Manufacturing":"Bulk","Foundry":"SemiBulk","Chemical Manufacturing":"Bulk","Finance":"SemiBulk","Insurance Agent":"SemiBulk"}
##SupplierIntensity
SupplierIntensity = {"Construction":"High","Jewellery":"Low","Restaurant":"Medium","Real Estate Agency":"Low","Equipment Hiring":"Low","Electronics":"Low","Manufacturing":"High","Clothes Manufacturing":"High",'Foundry':"Low","Chemical Manufacturing":"High","Finance":"Low","Insurance Agent":"Low"}
#Revenue_Size
RevenueSizeCatg = ['LOW','MEDIUM','HIGH']
#num_owners
NumQwners = [7,8,9]
#Business Size - small size business like a shop,medium like a wholeseller and large like biz bazzae
BusineeSizecatg = ['SMALL','MEDIUM','LARGE']
#Qwner History
HistoryQwner = ['YES','NO']
###Location
Location = ['OFFSHORE','INDIA']
###lOCATION OF COMPANY tax HAVEN 
TaxHavenLocation = ['YES','NO']
#Number of Accounts in the company
NumberOfAccounts = [1,2,3]
##Global Varibles for Transactions
##TransactionMode of the Transactions
SelectTransactionMode = ['CASH','CHEQUE','TRANSFER']
##AccountType
TypeAccount = ['Saving','Current']
#ExchangeType
ExchangeType = ['Local','Foreign','TaxHaven']
##Online or offline
OnlineOfflineMode = ['Online','Offline']
#Revenue Account
Income = ['Revenue']
#Expenditure
Expenditure = ['Salary','Utility','Supplier','ProfessionalServices','EntertainmentExpenses','PersonalExpenses']
#Current Year
c = datetime.datetime.now().year
##Generating Fake amounts
#FakeAmount1 = np.random.normal(50000,5000,10000)
#FakeAmount2 = np.random.normal(70000,2000,10000)
#FullFakeList = np.concatenate((FakeAmount1,FakeAmount2),axis=0)

##According to Me i need to Generate Different List for the Both Normal and Shell Company
FCurrencyN= []
ModeTransIncomingN = []
ModeTransOutgoingN = []
OnlineOfflineProbN = []

FCurrencyS = []
ModeTransIncomingS = []
ModeTransOutgoingS = []
OnlineOfflineProbS = []

TransactionNumber  = 0
##Global Parmeter but its value is referesed for each new 

def GenerateNormalDiscrete(FCurrencyP,ModeTransIncomingP,ModeTransOutgoingP,OnlineOfflineProbP):
	##Generate Transactional Variables  according to the Different Variables
	global FCurrencyN
	global ModeTransIncomingN
	global ModeTransOutgoingN
	global OnlineOfflineProbN
	
	
	##on avg i am expecting 20000 transactions.
	FCurrencyN = GenerateDiscreteValues(ExchangeType,FCurrencyP,20000)
	#cash,cheque,transfer
	ModeTransIncomingN = GenerateDiscreteValues(SelectTransactionMode,ModeTransIncomingP,20000)
	##cash,cheque,transfer
	ModeTransOutgoingN = GenerateDiscreteValues(SelectTransactionMode,ModeTransOutgoingP,20000)
	##Online and offline probability
	OnlineOfflineProbN = GenerateDiscreteValues(OnlineOfflineMode,OnlineOfflineProbP,20000)
	
def GenerateShellDiscrete(FCurrencyP,ModeTransIncomingP,ModeTransOutgoingP,OnlineOfflineProbP):
	global FCurrencyS
	global ModeTransIncomingS
	global ModeTransOutgoingS
	global OnlineOfflineProbS
	
	##Generate Transactional Variables  according to the Different Variables
	FCurrencyS = GenerateDiscreteValues(ExchangeType,FCurrencyP,20000)
	#cash,cheque,transfer
	ModeTransIncomingS = GenerateDiscreteValues(SelectTransactionMode,ModeTransIncomingP,20000)
	##cash,cheque,transfer
	ModeTransOutgoingS = GenerateDiscreteValues(SelectTransactionMode,ModeTransOutgoingP,20000)
	##Online and offline probability
	OnlineOfflineProbS = GenerateDiscreteValues(OnlineOfflineMode,OnlineOfflineProbP,20000)
	#print("Checking here because of some issue")
	#print(len(OnlineOfflineProbS))

##Income distribution of the Company
Type1 = np.random.normal(10000,2500,100000)
random.shuffle(Type1)
Type2 = np.random.normal(50000,10000,100000)
random.shuffle(Type2)
Type3 = np.random.normal(100000,10000,100000)
random.shuffle(Type3)
Type4 = np.random.normal(150000,15000,100000)
random.shuffle(Type4)
Type5 = np.random.normal(200000,20000,100000)
random.shuffle(Type5)

##For Expenditure Distribution
###Planning to add extra expenditure to increase the Annual Revenue of the Normal Company
##Offical Percentage Level Wise shell is more
#AddExtraExpenditure = np.random.normal(45,0.5,1000)
##Add adding the proportion of the company for which extra revenue is being addded to balance the scenario that 
##Normal company has much more revenue than the shell company
##ExtraExpenditureCompany = bernoulli.rvs(size=1000,p=0.85)	
#ExtraExpenditureCompany = list(ExtraExpenditureCompany)
##Percentage of the Expenditure curves for three different
RetailNormal= np.random.normal(0.10,0.005,100)
random.shuffle(RetailNormal)
RetailShell = np.random.normal(0.12,0.005,100)
random.shuffle(RetailShell)
MediumNormal = np.random.normal(0.060,0.004,100)
random.shuffle(MediumNormal)
MediumShell = np.random.normal(0.075,0.004,100)
random.shuffle(MediumShell)
HighNormal = np.random.normal(0.025,0.0025,100)
random.shuffle(HighNormal)
HighShell = np.random.normal(0.035,0.0025,100)
random.shuffle(HighShell)
##Generating Fake Amount of more than 100000
#FakeAmount = np.random.normal(95000,20000,100000)
##Generate Transaction Amount Curves
#'Salary','Utility','Supplier','ProfessionalServices','EntertainmentExpenses','PersonalExpenses'
## Target to keep shell More As compared to the Normal
UtilityExpNormal = np.random.normal(10000,10000,1000)
random.shuffle(UtilityExpNormal)
PersonalExpensesExpNormal = np.random.normal(50000,10000,1000)
random.shuffle(PersonalExpensesExpNormal)
ProfessionalServicesExpNormal = np.random.normal(80000,10000,1000)
random.shuffle(ProfessionalServicesExpNormal)
SupplierExpNormal = np.random.normal(120000,10000,1000)
random.shuffle(SupplierExpNormal)
SalaryExpNormal = np.random.normal(150000,10000,1000)
random.shuffle(SalaryExpNormal)
EntertainmentExpensesExpNormal = np.random.normal(180000,10000,1000)
random.shuffle(EntertainmentExpensesExpNormal)

"""UtilityExpShell = np.random.normal(30000,2000,100000)
random.shuffle(UtilityExpShell)
PersonalExpensesExpShell = np.random.normal(65000,5000,100000)
random.shuffle(PersonalExpensesExpShell)
ProfessionalServicesExpShell = np.random.normal(150000,10000,100000)
random.shuffle(ProfessionalServicesExpShell)
SupplierExpShell = np.random.normal(200000,10000,100000)
random.shuffle(SupplierExpShell)
SalaryExpShell = np.random.normal(220000,5000,100000)
random.shuffle(SalaryExpShell)
EntertainmentExpensesExpShell = np.random.normal(205000,1000,100000)
random.shuffle(EntertainmentExpensesExpShell)"""

UtilityExpShell = np.random.normal(12000,10000,1000)
random.shuffle(UtilityExpShell)
PersonalExpensesExpShell = np.random.normal(52000,10000,1000)
random.shuffle(PersonalExpensesExpShell)
ProfessionalServicesExpShell = np.random.normal(82000,10000,1000)
random.shuffle(ProfessionalServicesExpShell)
SupplierExpShell = np.random.normal(121000,10000,1000)
random.shuffle(SupplierExpShell)
SalaryExpShell = np.random.normal(151000,10000,1000)
random.shuffle(SalaryExpShell)
EntertainmentExpensesExpShell = np.random.normal(181000,10000,1000)
random.shuffle(EntertainmentExpensesExpShell)

##There are 2 types of Expenditure
#NormalExpenditure = np.random.normal(20,2,1000)
#ShellExpenditure = np.random.normal(18,2,1000)
#random.shuffle(NormalExpenditure)
#random.shuffle(ShellExpenditure)
##Method to Replace Random Fucntion
	
def get_random_key(a_huge_key_list) :
    L = len(a_huge_key_list)
    i = np.random.randint(0, L)
    return a_huge_key_list[i]
	
	
##This Bernoulli Distribution for Choosing Normal Company Which have large amount of the incoming Transaction
##Motive behind to increase average Numer of Incoming Transaction for the Normal Company


#LargeAmountTrans = bernoulli.rvs(size=1000,p=0.70)	
#LargeAmountTrans = list(LargeAmountTrans)	
###This Bernollui Distribution Describes 
data_bern = bernoulli.rvs(size=1000,p=0.40) 
###Analzed some logic ----- need 60:40 ratio is better
data_bern1 = list(data_bern)


## For the discrete variable lets go by taking binomail distributioN
"""def GenerateDiscreteValues(variableList,ProbOfSuccess,CompanyRatio):
	p = ProbOfSuccess
	size1 = int(float(CompanyRatio*0.10))
	size2 = CompanyRatio - size1
	points1 = np.random.choice(variableList,size2,ProbOfSuccess)
	if len(variableList) == 2:
		points2 = np.random.choice(variableList,size1,p=[0.5,0.5])
	else:
		points2 = np.random.choice(variableList,size1,p=[0.33,0.33,0.34])
	r2 = list(np.concatenate((points1, points2), axis=0))
	random.shuffle(r2) 
	return r2"""
	

"""def GenerateDiscreteValues(variableList,ProbOfSuccess,CompanyRatio):
    elements = variableList
    probabilities = ProbOfSuccess
    custm = np.random.choice(elements,CompanyRatio,p=probabilities)
    random.shuffle(custm) 
    return custm"""
	
##Method to add the randomness to the Discrete Dataset
def GenerateDiscreteValues(variableList,ProbOfSuccess,CompanyRatio):
    elements = variableList
    probabilities = ProbOfSuccess
    custm = np.random.choice(elements,CompanyRatio,p=probabilities)
    randomness = np.random.choice(variableList,1000)
    joinedList = [*custm, *randomness]
    random.shuffle(joinedList) 
    return joinedList
	
	
##Generate Header for Company Data and transaction data 
def GenerateHeader(Company):
	Flag = Company[0]
	with open('CompanyData_%s.csv' %Flag,mode = 'a') as Company_data,open('TransactionData_%s.csv' %Flag,mode = 'a') as TransactionData:
		CompanyObj = csv.writer(Company_data,delimiter = ",",quotechar='"', quoting=csv.QUOTE_MINIMAL)
		TransactionObj = csv.writer(TransactionData,delimiter = ",",quotechar='"', quoting=csv.QUOTE_MINIMAL)
		##Company Data
		CompanyObj.writerow(['AccountNumber','TypeBusiness','BusinessCategory','RevenueSize','BusinessSize','TotalEmployees','Annual_Revenue','TotalQwners','QwnerHistory','CompanyLocation','TaxHavenLocation','NumberOfAccounts','ExpenditureAmount','AnnualProfit','AccountType','PercentageExpenditure','Flag'])
		##Transaction Data
		TransactionObj.writerow(['AccountNumber','TransactionNumber','TransactionDate','TransactionCategory','TransactionMode','ExchangeType','OnlineOffline','TransactionAmount','PaymentTo','SourceAccount','DestinationAccount'])
##Calculate Total Number of Employees
def Esize(bsize):
	if bsize == 'SMALL':
		r1,r2 = 1,50
	elif bsize == 'MEDIUM':
		r1,r2 = 50,200
	else:r1,r2 = 200,500
	return random.randint(r1,r2)
#Calculate Annual Revenue of the Company
def AnnualRevenueCalculateNormal(rsize,Esize):
	if rsize == 'LOW':
		mean = 60000
	elif rsize == 'MEDIUM':
		mean = 110000
	else:
		mean = 600000
	std = 0.10 * mean
	X = np.random.normal(mean,std,1)
	Y = np.random.normal(0.1*X, 0.01*X,1)
	Z = np.random.normal(X,Y,1)
	Total_Revenue = int(float(12 * Esize * Z))
	return Total_Revenue
	
#Calculate Annual Revenue of the Company
def AnnualRevenueCalculateShell(rsize,Esize):
	if rsize == 'LOW':
		mean = 50000
	elif rsize == 'MEDIUM':
		mean = 100000
	else:
		mean = 500000
	std = 0.10 * mean
	X = np.random.normal(mean,std,1)
	Y = np.random.normal(0.1*X, 0.01*X,1)
	Z = np.random.normal(X,Y,1)
	Total_Revenue = int(float(12 * Esize * Z))
	return Total_Revenue
##Generate Client List
def GenerateClientList(Company,TempAnnualRevenue):
	RepeationAfterNormal = np.random.normal(10000,1000,1000)
	RepeationAfterShell = np.random.normal(11000,1000,1000)
	ClientList = []
	AccountID = 200001
	##AccountId for the client company
	if Company == 'Normal':
		RepeationAfter = get_random_key(RepeationAfterNormal)
	if Company == 'Shell':
		RepeationAfter = get_random_key(RepeationAfterShell)

	Mean = int(float(TempAnnualRevenue/RepeationAfter))
	std = Mean//10

	NumberOfClients = int(float(np.random.normal(Mean,std,1)))

	LowLimit = 200001

	HighLimit = 200001 + NumberOfClients + 1

	ClientList = np.arange(LowLimit,HighLimit)

	return ClientList

##Generate Destination list

DestinationList = np.arange(300001,300007)
random.shuffle(DestinationList)

#Generate Company Data
#Proportion of the Expenditure for Normal Company
##Proportion of the Expenditure for Normal Company
NormalExenditure = stats.skewnorm(40,28,20).rvs(100)
NormalExenditureValues = list(NormalExenditure)
ShellExenditure = stats.skewnorm(40,28.1,20).rvs(100)
ShellExenditureValues = list(ShellExenditure)

"""def CalculateExpenditure(Company,BusinessCategory):
	if Company == "Normal":
	#BusinessCatg = ['Retail','SemiBulk','Bulk']
		if BusinessCategory == "Retail":	
			percentage = np.random.normal(15,3,1)
		elif BusinessCategory == "SemiBulk":
			percentage = np.random.normal(12,2,1)
		else:
			percentage = np.random.normal(10,2,1)	
			
	if Company == "Shell":
		##Need to Some randomness in the Expenditure percentage
		percentage2 = np.random.normal(17,0.3,25)
		percentage1 = np.random.normal(96,1.8,100)
		percentageAll = [*percentage1, *percentage2]
		percentage = get_random_key(percentageAll)		
	return float(percentage)"""

def GenerateCompanyData(Company,CompanyRatio,AccountIDCompany,rsize,bsize,TQwner,QHistory,LocCompany,Taxhavenloc,NumberAccounts,SavingCurrent):
	##Generate List for the company data variables used different for Normal and Shell Company
	#print('Check Revenue size list')
	#print('CompanyRatio',CompanyRatio)
	#print("Company Name",Company)
	RevenueSizeList = GenerateDiscreteValues(RevenueSizeCatg,rsize,20000)
	#print('RevenueSizeList')
	#print(RevenueSizeList)
	AccEmpsizeList = GenerateDiscreteValues(BusineeSizecatg,bsize,20000)
	TotalQwnersList = GenerateDiscreteValues(NumQwners,TQwner,20000)
	QwnerHistoryList = GenerateDiscreteValues(HistoryQwner,QHistory,20000)
	CompanyLocationList = GenerateDiscreteValues(Location,LocCompany,20000)
	IslocationtaxHavenList = GenerateDiscreteValues(TaxHavenLocation,Taxhavenloc,20000)
	TotalAccountsList = GenerateDiscreteValues(NumberOfAccounts,NumberAccounts,20000)
	#print("check the value of the SavingCurrent",SavingCurrent)
	AccountTypeList = GenerateDiscreteValues(TypeAccount,SavingCurrent,20000)
			
	##Flag refers
	Flag = Company[0]
	##va
	#k = CompanyRatio - 1
	with open('CompanyData_%s.csv' %Flag,mode = 'a') as CompanyData:
		CompanyObj = csv.writer(CompanyData,delimiter = ",",quotechar='"', quoting=csv.QUOTE_MINIMAL)
		##If Company is Shell
		if Company == "Normal":
			sample = stats.skewnorm(15, 10, 15).rvs(CompanyRatio)
			MeanValues = list(sample)
		
		##Planning to divide code into Two different Parts
		CountDivision = 0
		if Company == "Normal":
			Parts1 = CompanyRatio//2
			Parts2 = Parts1 + 1
			
		for i in range(CompanyRatio):
			print("company number is",i)
			global TransactionNumber
			global CountTruePositive
			TransactionNumber = 1001
			#print("company number is",i)
			#print("Value of k",k)
			##Added time to check the start of one Account
			t0 = time.time()  # start time
			BusinessType = random.choice(Business)
			##Supplier Intensity
			SupplierValue = SupplierIntensity[BusinessType]
			##Retail,semibulk,bulk
			BusinessCategory = BusinessSize[BusinessType]
			#LOW,MEDIUM,HIGH   		
			RevenueSize = get_random_key(RevenueSizeList)	
			##SMALL,MEDIUM,LARGE
			AccEmpsize = get_random_key(AccEmpsizeList)
			###small (1-50),medium(50-200),large(200-500)
			EmployeeSize = Esize(AccEmpsize)
			##Annual revenue of the compnay according to revenue size and employee size
			if Company == "Normal":
				AnnualRevenue = AnnualRevenueCalculateNormal(RevenueSize,EmployeeSize)
			elif Company == "Shell":
				AnnualRevenue = AnnualRevenueCalculateShell(RevenueSize,EmployeeSize)
			##Number of qwner of the company
			TotalQwners = get_random_key(TotalQwnersList)
			#History of the qwner
			QwnerHistory = get_random_key(QwnerHistoryList)
			##Company Location
			CompanyLocation = get_random_key(CompanyLocationList)
			if CompanyLocation == 'OFFSHORE':
				##If offshore company then tax haven else it cant be taxhaven
				IslocationtaxHaven = get_random_key(IslocationtaxHavenList)
			else:
				IslocationtaxHaven = 0
			###Temporaray variable generated to assign Annual Revenue value
			TempAnnualRevenue = AnnualRevenue
			##total Number of accounts
			TotalAccounts = get_random_key(TotalAccountsList)
			##Check Company open saving or current account
			AccountType = get_random_key(AccountTypeList)

			#k = k - 1
			##Calculate percentage of Expiduture
			if Company == "Normal":
				PercentageExpenditure = get_random_key(NormalExenditureValues)
				PercentageWise = PercentageExpenditure
			else:
				PercentageExpenditure = get_random_key(ShellExenditureValues)	
			#PercentageExpenditure = CalculateExpenditure(Company,BusinessCategory)
			#print(PercentageExpenditure)
			if PercentageExpenditure > 100:
				PercentageWise = 100 - PercentageExpenditure
			#print("PercentageExpenditure",PercentageExpenditure)
			ExpenditureAmount = int(float(TempAnnualRevenue * np.floor(PercentageExpenditure)))/100
			
			##Here for the Normal Company I need to add some extra revenue
			#ChooseExtraExpRequired = get_random_key(ExtraExpenditureCompany)
			#if Company == "Normal" and ChooseExtraExpRequired == 1:
				##I am here applying bernoulli Distributiion
				##I am giving the Extra expenditure to the 60% of the Company in the proportion of the 8 - 10%)
			#	PercentageExtraExp = get_random_key(AddExtraExpenditure)
			#	ExtraExp = int(float(TempAnnualRevenue * np.floor(PercentageExtraExp)))/100
			#	ExpenditureAmount = ExpenditureAmount + ExtraExp				
			"""if Company == "Normal":
				#print("Chirag")
				##Define the Method to calculate the Percentage of the expenditure
				#PercentageExpenditure = CalculateExpenditure(Company,BusinessCategory)
				#NormalValue = get_random_key(NormalExpenditure)
				#print("check exp",NormalValue)
				ExpenditureAmount = int(float(TempAnnualRevenue * np.floor(get_random_key(NormalExpenditure)))/100)
			elif Company == "Shell":
				#print("Kriti")
				ShellValue = get_random_key(ShellExpenditure)
				#print("check shell exp",ShellValue)				
				ExpenditureAmount = int(float(TempAnnualRevenue * np.floor(get_random_key(ShellExpenditure)))/100)"""
			
				
			Profit = int(float(AnnualRevenue - ExpenditureAmount))		
		
			CompanyObj.writerow([AccountIDCompany,BusinessType,BusinessCategory,RevenueSize,AccEmpsize,EmployeeSize,AnnualRevenue,TotalQwners,QwnerHistory,CompanyLocation,IslocationtaxHaven,TotalAccounts,ExpenditureAmount,Profit,AccountType,PercentageExpenditure,Flag])
			
				
			if Company == "Normal":
				#print("check presence of going here")
				#LargeAmountTransvalue = get_random_key(LargeAmountTrans)
				RangeShell = get_random_key(MeanValues)
				#GenerateCreditTransaction(Company,AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,SavingCurrentAccount,"N")
				#GenerateDebitTransaction(Company,AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,SavingCurrentAccount,"N")
				##Here we are adding because we want more number of false negative in qur dataset and want to decrease accuracy
				#if CountDivision <= Parts1 :
				if RangeShell >=120:	
					CountTruePositive = CountTruePositive + 1	
					#FakeRevenue = 0.03 * TempAnnualRevenue
					#RevenueLeft = TempAnnualRevenue - FakeRevenue
					#GenerateFakeCreditTransactions("Shell",AccountIDCompany,FakeRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN,1)
					GenerateCreditTransaction("Shell",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS,1)
					GenerateDebitTransaction("Shell",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS,1)
				else:
					##Planning to Generate Some percentage of the Large Amount of the Transactions for the Normal Company 
					##To increase average Number of Incoming Trasacrtions
					##Using Bernoulli Distribution assuming 40% of transaction to increase in the large amount of incoming transactions
					#if LargeAmountTransvalue == 1:
						#FakeRevenue = 0.40 * TempAnnualRevenue
						#RevenueAfterFake = TempAnnualRevenue - FakeRevenue 				
						#GenerateFakeCreditTransactions("Normal",AccountIDCompany,FakeRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN,1)
					
					#FakeRevenue = 0.05 * TempAnnualRevenue
					#RevenueLeft = TempAnnualRevenue - FakeRevenue
					#GenerateFakeCreditTransactions("Normal",AccountIDCompany,FakeRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN,1)				
					GenerateCreditTransaction("Normal",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN,1)
					GenerateDebitTransaction("Normal",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN,1)
					#elif LargeAmountTransvalue == 0:
						#GenerateCreditTransaction("Normal",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN,1)
						#GenerateDebitTransaction("Normal",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN,1)
						
				"""else:
					if RangeShell >=50:	
						CountTruePositive = CountTruePositive + 1					
						GenerateCreditTransaction("Shell",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS,2)
						GenerateDebitTransaction("Shell",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS,2)
					else:
						GenerateCreditTransaction("Normal",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN,2)
						GenerateDebitTransaction("Normal",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"N",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN,2)"""
				

			elif Company == "Shell":
				##This Means there is 20% chance of Shell Company Transaction to behave like the Normal Company
				value = get_random_key(data_bern1)
				if value == 1:
					CountTruePositive = CountTruePositive + 1	
					#FakeRevenue = 0.03 * TempAnnualRevenue
					#RevenueLeft = TempAnnualRevenue - FakeRevenue
					#GenerateFakeCreditTransactions("Shell",AccountIDCompany,FakeRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN,1)
					GenerateCreditTransaction("Shell",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS,1)
					GenerateDebitTransaction("Shell",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS,1)
				else:
					ShellCompanyRevenue = TempAnnualRevenue * 0.45
					NormalCompanyRevenue = TempAnnualRevenue * 0.65
					GenerateCreditTransaction("Normal",AccountIDCompany,NormalCompanyRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN,1)
					GenerateCreditTransaction("Shell",AccountIDCompany,ShellCompanyRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS,1)
					GenerateDebitTransaction("Normal",AccountIDCompany,NormalCompanyRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN,1)
					GenerateDebitTransaction("Shell",AccountIDCompany,ShellCompanyRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS,1)			
				##I am here handling the case where the transactions of the shell company is treated as mostly the same as the normal company 
				#ShellCompanyRevenue = TempAnnualRevenue * 0.20
				#NormalCompanyRevenue = TempAnnualRevenue * 0.80	
				#GenerateCreditTransaction("Normal",AccountIDCompany,NormalCompanyRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN)
				#GenerateCreditTransaction("Shell",AccountIDCompany,ShellCompanyRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS)
				#GenerateDebitTransaction("Normal",AccountIDCompany,NormalCompanyRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyN,ModeTransIncomingN,ModeTransOutgoingN,OnlineOfflineProbN)
				#GenerateDebitTransaction("Shell",AccountIDCompany,ShellCompanyRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS)
				#GenerateCreditTransaction("Shell",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS)
				#GenerateDebitTransaction("Shell",AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,"S",FCurrencyS,ModeTransIncomingS,ModeTransOutgoingS,OnlineOfflineProbS)
			AccountIDCompany = AccountIDCompany + 1
			# the code to time goes here
			t1 = time.time() # end time
		#print("Total Number of true positive",CountTruePositive)
			
##Generate Income transaction
def generateIncome(Company,Business_Category,Revenue_Size):
	##Large Amount of Incoming transactions for Normal company
	#BusinessCatg = ['Retail','SemiBulk','Bulk']
	#RevenueSizeCatg = ['LOW','MEDIUM','HIGH']
	p = []	
	##To make the choice between Five curve acc to the Binomail distribution
	IncomeType = ['Type1','Type2','Type3','Type4','Type5']

	DictValues = {'LOW':0,'MEDIUM':1,'HIGH':2}
	
	if Business_Category == "Retail":
		DictValues = {'LOW':0,'MEDIUM':1,'HIGH':2}
	elif Business_Category == "SemiBulk":
		DictValues = {'LOW':3,'MEDIUM':4,'HIGH':5}
	elif Business_Category == "Bulk":
		DictValues = {'LOW':6,'MEDIUM':7,'HIGH':8}

	i = DictValues[Revenue_Size]

	##Average Number of Incoming Transactions are more for the Normal company
	p_value_Normal = np.zeros((5, 5))
	p_value_Normal = ([[0.20,0.18,0.21,0.19,0.22],##Retail + low
					   [0.20,0.19,0.21,0.20,0.20],###Retail + Medium
					   [0.20,0.19,0.20,0.20,0.21],#Retail + high
					   [0.21,0.21,0.20,0.19,0.19],##SemiBulk + low
					   [0.19,0.20,0.20,0.20,0.21],##SemiBulk + Medium
					   [0.21,0.20,0.21,0.19,0.19],##SemiBulk + high
					   [0.19,0.18,0.21,0.21,0.21],##BULK + low
					   [0.20,0.20,0.21,0.20,0.19],##BULK + Meduim
					   [0.20,0.19,0.21,0.19,0.21]])##Bulk + high

	##small amount of incoming transaction for the shell company is more
	p_value_Shell = np.zeros((5, 5))  
	##Will verify if there is randomness required in this case
	##Target to decrease small amount of incoming transaction for the shell company
	p_value_Shell = ([[0.20,0.20,0.20,0.20,0.20],##Retail + low
					   [0.20,0.20,0.20,0.20,0.20],###Retail + Medium
					   [0.20,0.20,0.20,0.20,0.20],#Retail + high
					   [0.20,0.20,0.20,0.20,0.20],##SemiBulk + low
					   [0.20,0.20,0.20,0.20,0.20],##SemiBulk + Medium
					   [0.20,0.20,0.20,0.20,0.20],##SemiBulk + high
					   [0.20,0.20,0.20,0.20,0.20],##BULK + low
					   [0.20,0.20,0.20,0.20,0.20],##BULK + Meduim
					   [0.20,0.20,0.20,0.20,0.20]])##Bulk + high		 
				   
	if Company == "Normal":
		p = p_value_Normal[i]
	else:
		p = p_value_Shell[i]
		
	Amount = Type1 * p[0] + Type2 * p[1] + Type3 * p[2] + Type4 * p[3] + Type5 * p[4]
	#print("Amount",Amount)
	#TransactionAmount =  random.choice(Amount)
	return Amount
	#return int(float(Amount))

##Generate Credit Transaction
def GenerateCreditTransaction(Company,AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,Type,FCurrencyP,ModeTransIncomingP,ModeTransOutgoingP,OnlineOfflineProbP,SheetNumber):
	
	##Just for the Purpose of testing
	#print("just for the purpose of testing")
	#print(OnlineOfflineProbP)
	global TransactionNumber
	
	Flag = Type
	##Generate Client List
	ClientList = GenerateClientList(Company,TempAnnualRevenue)
	random.shuffle(ClientList)
	
	
	AmountGenerated = generateIncome(Company,BusinessCategory,RevenueSize)

	##Here We are opening the file according to the Type of the file
	with open('TransactionData_%s.csv'%Flag,mode = 'a') as TransactionData:
		Transaction_obj = csv.writer(TransactionData,delimiter = ",",quotechar='"', quoting=csv.QUOTE_MINIMAL)
		##Incoming Distribution
		IncomingDistribution = 0
		#TransactionNumber = 1001
		##Loop time Annual Revenue is exhausted
		#cnt = 9999
		while(TempAnnualRevenue >= IncomingDistribution): 
			#if cnt == 1:
			#	cnt = 9999
			#count = 99999
			##Flag to identify Normal/Shell 
			#TransactionNumber = 1001			
			##Transaction Date
			start_dt = datetime.date.today().replace(day=1, month=1).toordinal()
			end_dt = datetime.date.today().toordinal()
			TransactionDate = datetime.date.fromordinal(random.randint(start_dt, end_dt))
			##Select Category of the Transaction --- Debit and the credit
			DebitCredit = "Credit"
			##For the purpose of testing chutiyann tym
			
			TransactionMode = get_random_key(ModeTransIncomingP)
			TransactionType = get_random_key(FCurrencyP)
			
			#TransactionMode = ModeTransIncomingP[cnt]
			#TransactionType = FCurrencyP[cnt]
			if TransactionMode == "TRANSFER":
				##Online And Offline Mode
				OnlineOffline = get_random_key(OnlineOfflineProbP)
				#OnlineOffline = OnlineOfflineProbP[cnt]
				#OnlineOffline = OnlineOfflineProb[count]
				#OnlineOffline = " ".join(get_random_key(OnlineOfflineMode,1,p = OnlineOfflineProb))
			else:
				OnlineOffline = "Offline"
			#print("value of cnt in credit transaction",cnt)
			#cnt = cnt - 1
			##TransactionAmount to generate income transaction
			#TransactionAmount = generateIncome(Company,BusinessCategory,RevenueSize)
			TransactionAmount = int(float(get_random_key(AmountGenerated)))
			#PaymentTo
			PaymentTo = "Revenue"
			##Genrate Source Account and the Destination Account
			SourceAccount = random.choice(ClientList)
			##Destination Account
			DestinationAccount = AccountIDCompany
			##Write obj   
			if TransactionAmount > 0:
				Transaction_obj.writerow([AccountIDCompany,TransactionNumber,TransactionDate,DebitCredit,TransactionMode,TransactionType,OnlineOffline,TransactionAmount,PaymentTo,SourceAccount,DestinationAccount])
				#Trans Number is incremented
				TransactionNumber = TransactionNumber + 1
				IncomingDistribution = IncomingDistribution + TransactionAmount
				TempAnnualRevenue = TempAnnualRevenue - TransactionAmount
				#count = count - 1


##Generate Expenditure
def GenerateExpenditure(Company,BusinessCategory,RevenueSize,PaymentTo,MonthlyExpenditure,SupplierValue):
	#BusinessCatg = ['Retail','SemiBulk','Bulk']
	#RevenueSizeCatg = ['LOW','MEDIUM','HIGH']
	
	##My Assumption
	#Expenditure = ['Salary','Utility','Supplier','ProfessionalServices','EntertainmentExpenses','PersonalExpenses']
	
	if Company == "Normal":
		if PaymentTo == 'Salary':
			SalaryDistribution = {'Retail':get_random_key(RetailNormal),'SemiBulk':get_random_key(MediumNormal),'Bulk':get_random_key(HighNormal)}
			TransactionAmount = MonthlyExpenditure * SalaryDistribution[BusinessCategory]

		if PaymentTo == 'Supplier':
			SupplierDistribution = {'LOW':get_random_key(RetailNormal),'MEDIUM':get_random_key(MediumNormal),'HIGH':get_random_key(HighNormal)}
			TransactionAmount = MonthlyExpenditure * SupplierDistribution[RevenueSize]

		if PaymentTo == 'Utility':
			UtilityDistribution = {'Retail':get_random_key(RetailNormal),'SemiBulk':get_random_key(MediumNormal),'Bulk':get_random_key(HighNormal)}
			TransactionAmount = MonthlyExpenditure * UtilityDistribution[BusinessCategory]

		if PaymentTo == 'ProfessionalServices':
			ProfessionalDistribution = {'LOW':get_random_key(RetailNormal),'MEDIUM':get_random_key(MediumNormal),'HIGH':get_random_key(HighNormal)}
			TransactionAmount = MonthlyExpenditure * ProfessionalDistribution[RevenueSize]

		if PaymentTo == "EntertainmentExpenses":
			RandomDistribution = {'LOW':get_random_key(RetailNormal),'MEDIUM':get_random_key(MediumNormal),'HIGH':get_random_key(HighNormal)}
			TransactionAmount = MonthlyExpenditure * RandomDistribution[RevenueSize]

		if PaymentTo == "PersonalExpenses":
			BulkyDistribution = {'LOW':get_random_key(RetailNormal),'MEDIUM':get_random_key(MediumNormal),'HIGH':get_random_key(HighNormal)}
			TransactionAmount = MonthlyExpenditure * BulkyDistribution[RevenueSize]

	elif Company == "Shell":

		if PaymentTo == 'Salary':
			SalaryDistribution = {'Retail':get_random_key(RetailShell),'SemiBulk':get_random_key(MediumShell),'Bulk':get_random_key(HighShell)}
			TransactionAmount = MonthlyExpenditure * SalaryDistribution[BusinessCategory]

		if PaymentTo == 'Supplier':
			SupplierDistribution = {'LOW':get_random_key(RetailShell),'MEDIUM':get_random_key(MediumShell),'HIGH':get_random_key(HighShell)}
			TransactionAmount = MonthlyExpenditure * SupplierDistribution[RevenueSize]

		if PaymentTo == 'Utility':
			UtilityDistribution = {'Retail':get_random_key(RetailShell),'SemiBulk':get_random_key(MediumShell),'Bulk':get_random_key(HighShell)}
			TransactionAmount = MonthlyExpenditure * UtilityDistribution[BusinessCategory]

		if PaymentTo == 'ProfessionalServices':
			ProfessionalDistribution = {'LOW':get_random_key(RetailShell),'MEDIUM':get_random_key(MediumShell),'HIGH':get_random_key(HighShell)}
			TransactionAmount = MonthlyExpenditure * ProfessionalDistribution[RevenueSize]

		if PaymentTo == "EntertainmentExpenses":
			RandomDistribution = {'LOW':get_random_key(RetailShell),'MEDIUM':get_random_key(MediumShell),'HIGH':get_random_key(HighShell)}
			TransactionAmount = MonthlyExpenditure * RandomDistribution[RevenueSize]

		if PaymentTo == "PersonalExpenses":
			BulkyDistribution = {'LOW':get_random_key(RetailShell),'MEDIUM':get_random_key(MediumShell),'HIGH':get_random_key(HighShell)}
			TransactionAmount = MonthlyExpenditure * BulkyDistribution[RevenueSize]

	return int(float(TransactionAmount))


### Genrate Last_day of month
##It is the list of the last day of the month
LastDay = []

def last_day_of_month(any_day):
	next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
	return next_month - datetime.timedelta(days=next_month.day)

for month in range(1, 13):
	a = last_day_of_month(datetime.date(2019, month, 1))
	LastDay.append(a.strftime('%m/%d/%Y'))
	
def GenerateTransactionAmount(Company,PaymentTo):
	##'Salary','Utility','Supplier','ProfessionalServices','EntertainmentExpenses','PersonalExpenses
	##if company is normal
	##Just for Assignment
	
	if Company == "Normal":
		if PaymentTo == "Salary":
			#print("Chirag Check")
			#print(SalaryExpNormal.shape)
			#TransactionAmount = get_random_key(SalaryExpNormal)
			return SalaryExpNormal
			
		elif PaymentTo == "Utility":
			#TransactionAmount = get_random_key(UtilityExpNormal)
			return UtilityExpNormal
			
		elif PaymentTo == "Supplier":
			return SupplierExpNormal
			#TransactionAmount = get_random_key(SupplierExpNormal)
		
		elif PaymentTo == "ProfessionalServices":
			return ProfessionalServicesExpNormal
			#TransactionAmount = get_random_key(ProfessionalServicesExpNormal)
		
		elif PaymentTo == "EntertainmentExpenses":
			return EntertainmentExpensesExpNormal
			#TransactionAmount = get_random_key(EntertainmentExpensesExpNormal)
		
		elif PaymentTo == "PersonalExpenses":	
			return PersonalExpensesExpNormal
			#TransactionAmount = get_random_key(PersonalExpensesExpNormal)
			
		else:
			return PersonalExpensesExpNormal
			#TransactionAmount = get_random_key(PersonalExpensesExpNormal)
		

	elif Company == "Shell":
		if PaymentTo == "Salary":
			return SalaryExpShell
			#TransactionAmount = get_random_key(SalaryExpShell)
			
		elif PaymentTo == "Utility":
			return UtilityExpShell
			#TransactionAmount = get_random_key(UtilityExpShell)
			
		elif PaymentTo == "Supplier":
			return SupplierExpShell
			#TransactionAmount = get_random_key(SupplierExpShell)
		
		elif PaymentTo == "ProfessionalServices":
			return ProfessionalServicesExpShell
			#TransactionAmount = get_random_key(ProfessionalServicesExpShell)
		
		elif PaymentTo == "EntertainmentExpenses":
			return EntertainmentExpensesExpShell
			#TransactionAmount = get_random_key(EntertainmentExpensesExpShell)
		
		elif PaymentTo == "PersonalExpenses":	
			return PersonalExpensesExpShell
			#TransactionAmount = get_random_key(PersonalExpensesExpShell)
		
		else:
			return PersonalExpensesExpNormal
			#TransactionAmount = get_random_key(PersonalExpensesExpNormal)
		
	#return TransactionAmount

#This method is used to generate random date for the given year,month
def randomdate(year, month):
	dates = calendar.Calendar().itermonthdates(year, month)
	return random.choice([date for date in dates if date.month == month])

def GenerateTransactions(Company,PaymentTo,PercentageAmount,FCurrencyP,ModeTransIncomingP,ModeTransOutgoingP,OnlineOfflineProbP,AccountIDCompany,i,Type):
	Flag = Type
	with open('TransactionData_%s.csv' %Flag,mode = 'a') as TransactionData:
		Transaction_obj = csv.writer(TransactionData,delimiter = ",",quotechar='"', quoting=csv.QUOTE_MINIMAL)
		###Intermediate Variable
		OutgoingAmountType = PercentageAmount
		OutgoingCheck = 0
		#count = 99999
		#TransactionNumber = 1001
		#cnt = 999
		global TransactionNumber
		##In this I am generating the values according to the type of the Expenditure
		SubsetInformation = GenerateTransactionAmount(Company,PaymentTo)
		#print(SubsetInformation)
		##bewkoff awrat u r choosing i according to the month		
		while OutgoingAmountType >= OutgoingCheck:
			#if cnt == 1:
			#	cnt = 999
			##Amount spent on the salary for the given month
			##Transaction Date
			if PaymentTo == 'Salary':
				TransactionMode = "TRANSFER"
				#print("Again Baby check", LastDay)
				#print("WHAT IS VALUE OF  I ",i)
				TransactionDate = LastDay[i-1]
			else:
				#print("Value of i",i)
				TransactionDate = randomdate(c,i)
				
			##Only to handle the case for each month last day to be selected
			#i = i + 1
			##Select Category of the Transaction --- Debit and the credit
			DebitCredit = "Debit"
			## Select Exchange Type -- local,foreign,Tax-haven
			#print("What is the value of count coming to me : ",count)
			#print(FCurrency)
			#print(len(FCurrency))
			#print("count is",count)
			##FCurrencyP,ModeTransIncomingP,ModeTransOutgoingP,OnlineOfflineProbP
			##For the purpose of testing chutiyann tym
			TransactionMode = get_random_key(ModeTransIncomingP)
			#TransactionMode = ModeTransIncomingP[cnt]
			TransactionType = get_random_key(FCurrencyP)
			#TransactionType = FCurrencyP[cnt]
			if TransactionMode == "TRANSFER":
				##Online And Offline Mode				
				OnlineOffline = get_random_key(OnlineOfflineProbP)
				#OnlineOffline = OnlineOfflineProbP[cnt]
				#OnlineOffline = OnlineOfflineProb[count]
				#OnlineOffline = " ".join(get_random_key(OnlineOfflineMode,1,p = OnlineOfflineProb))
			else:
				OnlineOffline = "Offline"
			#print("check value of cnt",cnt)
			#cnt = cnt - 1
			#Tranaction Amount
			#TransactionAmount = GenerateTransactionAmount(Company,PaymentTo)
			TransactionAmount = int(float(get_random_key(SubsetInformation)))
			#print(TransactionAmount)
			##Genrate Source Account and the Destination Account
			SourceAccount = AccountIDCompany
			##Destination Account
			DestinationAccount = random.choice(DestinationList)
			
			if TransactionAmount >0:
				##Generate Expenditure
				Transaction_obj.writerow([AccountIDCompany,TransactionNumber,TransactionDate,DebitCredit,TransactionMode,TransactionType,OnlineOffline,TransactionAmount,PaymentTo,SourceAccount,DestinationAccount])

				TransactionNumber = TransactionNumber + 1

				OutgoingCheck = OutgoingCheck + TransactionAmount

				OutgoingAmountType = OutgoingAmountType - TransactionAmount
				##Change Count valeue
				#count = count - 1	
				
##This Method is used to Generate Fake Transactions
"""def GenerateFakeCreditTransactions(Company,AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,Type,FCurrencyP,ModeTransIncomingP,ModeTransOutgoingP,OnlineOfflineProbP,SheetNumber):
	
	##Just for the Purpose of testing
	#print("just for the purpose of testing")
	#print(OnlineOfflineProbP)
	
	
	global TransactionNumber
	
	Flag = Type
	##Generate Client List
	ClientList = GenerateClientList(Company,TempAnnualRevenue)
	random.shuffle(ClientList)
	

	#AmountGenerated = generateIncome(Company,BusinessCategory,RevenueSize)


	##Here We are opening the file according to the Type of the file
	with open('TransactionData_%s.csv'%Flag,mode = 'a') as TransactionData:
		Transaction_obj = csv.writer(TransactionData,delimiter = ",",quotechar='"', quoting=csv.QUOTE_MINIMAL)
		##Incoming Distribution
		IncomingDistribution = 0
		#TransactionNumber = 1001
		##Loop time Annual Revenue is exhausted
		#cnt = 9999
		while(TempAnnualRevenue >= IncomingDistribution): 
			#if cnt == 1:
			#	cnt = 9999
			#count = 99999
			##Flag to identify Normal/Shell 
			#TransactionNumber = 1001			
			##Transaction Date
			start_dt = datetime.date.today().replace(day=1, month=1).toordinal()
			end_dt = datetime.date.today().toordinal()
			TransactionDate = datetime.date.fromordinal(random.randint(start_dt, end_dt))
			##Select Category of the Transaction --- Debit and the credit
			DebitCredit = "Credit"
			##For the purpose of testing chutiyann tym
			
			TransactionMode = get_random_key(ModeTransIncomingP)
			TransactionType = get_random_key(FCurrencyP)
			
			#TransactionMode = ModeTransIncomingP[cnt]
			#TransactionType = FCurrencyP[cnt]
			if TransactionMode == "TRANSFER":
				##Online And Offline Mode
				OnlineOffline = get_random_key(OnlineOfflineProbP)
				#OnlineOffline = OnlineOfflineProbP[cnt]
				#OnlineOffline = OnlineOfflineProb[count]
				#OnlineOffline = " ".join(get_random_key(OnlineOfflineMode,1,p = OnlineOfflineProb))
			else:
				OnlineOffline = "Offline"
			#print("value of cnt in credit transaction",cnt)
			#cnt = cnt - 1
			##TransactionAmount to generate income transaction
			#TransactionAmount = generateIncome(Company,BusinessCategory,RevenueSize)
			TransactionAmount = int(float(get_random_key(FullFakeList)))
			#PaymentTo
			PaymentTo = "Revenue"
			##Genrate Source Account and the Destination Account
			SourceAccount = random.choice(ClientList)
			##Destination Account
			DestinationAccount = AccountIDCompany
			##Write obj   
			if TransactionAmount > 0:
				Transaction_obj.writerow([AccountIDCompany,TransactionNumber,TransactionDate,DebitCredit,TransactionMode,TransactionType,OnlineOffline,TransactionAmount,PaymentTo,SourceAccount,DestinationAccount])
				#Trans Number is incremented
				TransactionNumber = TransactionNumber + 1
				IncomingDistribution = IncomingDistribution + TransactionAmount
				TempAnnualRevenue = TempAnnualRevenue - TransactionAmount
				#count = count - 1"""
		
	
##Lets Modify Method for Generation of Monthly Expenditure
def GenerateDebitTransaction(Company,AccountIDCompany,TempAnnualRevenue,BusinessCategory,RevenueSize,ExpenditureAmount,SupplierValue,Type,FCurrencyP,ModeTransIncomingP,ModeTransOutgoingP,OnlineOfflineProbP,SheetNumber):
	Flag = Type
	##Open Transaction File for Updation

	#Percentage Of Monthly Expenditure
	MonthlyExpenditure = int(ExpenditureAmount/12)
		
	for i in range(1,13):
		#print("every time check the value of i", i )
		J = 0
		#print("Company name",Company)
		##Assign Monthly Expenditure for every time
		PerMonthExpenditure = MonthlyExpenditure
		OutgoingMoney = 0
		while PerMonthExpenditure >  OutgoingMoney and J < 6:
			#print("Value of J here is",J)
			##According to the type of Payment Choose the Expenditure Percentage
			PaymentTo = Expenditure[J]
			#print("check to whom to payment",PaymentTo)
			#Percentage Amount for the Each type of Expenditure
			PercentageAmount = GenerateExpenditure(Company,BusinessCategory,RevenueSize,PaymentTo,MonthlyExpenditure,SupplierValue)	
			#print("Percentage Amount is",PercentageAmount)
			##Generate Transactions for the Given Percentage Amount			
			GenerateTransactions(Company,PaymentTo,PercentageAmount,FCurrencyP,ModeTransIncomingP,ModeTransOutgoingP,OnlineOfflineProbP,AccountIDCompany,i,Type)
			##Here we are Substracting from the Total Monthly  Expenditure
			OutgoingMoney = OutgoingMoney + PercentageAmount
			PerMonthExpenditure = PerMonthExpenditure - PercentageAmount
			J = J + 1
		
		RemainingExpenditure = PerMonthExpenditure
		if RemainingExpenditure > 0:
			GenerateTransactions(Company,"Random",RemainingExpenditure,FCurrencyP,ModeTransIncomingP,ModeTransOutgoingP,OnlineOfflineProbP,AccountIDCompany,i,Type)
		
