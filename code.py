# Libraries
import sys
import os
import glob
import warnings
sys.path.insert(0, '../lib')  # noqa
import numpy as np
import pandas as pd
#import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
import pdb
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools
import csv
np.random.seed(1)
plt.style.use('ggplot')
import xgboost as xgb
from numpy import loadtxt
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import sort
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model
import lightgbm as lgb
import json
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import math




# Opening all the dataset one by one (Spectrum)

data_str = open('E:/My Thesis/d4jpartitopned/S1.json').read()
df_X1 = pd.read_json(data_str, orient='records')              #Repeat the same two lines for the total number of faulty lines
..
..
..
..
data_str = open('E:/My Thesis/d4jpartitopned/S357.json').read()
df_X357 = pd.read_json(data_str, orient='records')


# Opening all the dataset one by one (Result (Faulty/Non-Faulty)

data_str2 = open('E:/My Thesis/d4jpartitopned/Dy1.json').read()
df_y1 = pd.read_json(data_str2, orient='records')             #Repeat the same two lines for the total number of faulty lines
..
..
..
..
data_str2 = open('E:/My Thesis/d4jpartitopned/Dy357.json').read()
df_y357 = pd.read_json(data_str2, orient='records')

# Combining the Spectrum and their outputs together
df1 = pd.concat([df_X1,df_y1],axis=1)
..
..
..
..
df357 = pd.concat([df_X357,df_y357],axis=1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This code is used to perform the pre-experiment study on 30 faults
# The aim of this pre-experiment was to shortlist the metrics that can be combined 
# This is repeated for 30 times while randomly changing the faults in dfLr
# This experinemt was run for each of the studied metrics
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dfLr = [df1,df2,df3,df4,df5,df27,df28,df29,df30,df31,df160,df161,df162,df163,df164,df225,df226,df227,df228,df229,df331,df332,df333,df334,df335,dfs1,dfs2,dfs3,dfs4,dfs5]
file_list=[
'C:/My Thesis/Paper two modification/New Experiment/Linear/fault1.json',
..
..
..
'C:/My Thesis/Paper two modification/New Experiment/Linear/fault25.json',]

for j in range(len(file_list)):
    l1=dfLr[j]
    l1.columns=['EP', 'EF', 'NP', 'NF', 'target']
    
    
    def di(x, y):
        return 0 if np.any(y == 0) else x / y
    def gp_div(x,y):
        try:
            return x/y
        except ZeroDivisionError:
            return 0
    
	# List of Existing formulas
    l1['OP2'] = l1['EF'] - (l1['EP']/(l1['EP']+l1['NP']+1))
    l1['ER1a'] = l1.apply(lambda x: -1.000 if (x[1] < (x[1] + x[3])) else x[2], axis=1)
    l1['Tar'] = np.divide((np.divide(l1['EF'],(l1['EF']+l1['NF']))),(np.divide(l1['EF'],(l1['EF']+l1['NF']))+np.divide(l1['EP'],(l1['EP']+l1['NP']))))
    l1['Och1'] = np.divide(l1['EF'],(np.sqrt(l1['EF']+l1['NF'])*(l1['EF']+l1['EP'])))
    l1['Och2'] = gp_div((l1['EF']*l1['NP']),((l1['EF']+l1['EP'])*(l1['NP']+l1['NF'])*(l1['EF']+l1['NF'])*(l1['EP']+l1['NP'])))
    l1['Amp'] = abs((l1['EF']/(l1['EF']+l1['NF']))-(l1['EP']/(l1['EP']+l1['NP'])))
    l1['Jac'] = l1['EF']/(l1['EF']+l1['EP']+l1['NF'])
    l1['D2'] = l1['EF']**2/(l1['EP']+l1['NF'])
    l1['GP2'] = 2*(l1['EF']+np.sqrt(l1['NP'])+np.sqrt(l1['EP']))
    l1['GP3'] = np.sqrt(abs(l1['EF']**2-np.sqrt(l1['EP'])))
    l1['GP13'] = l1['EF']*(1+(1/(2*l1['EP']+l1['EF'])))
    l1['GP19'] = l1['EF']*(np.sqrt(abs(l1['EP']-l1['EF']+l1['NF']-l1['NP'])))     
    l1['Wong1'] = l1['EF']
    l1['Wong2'] = (l1['EF']-l1['EP'])
    l1['Wong3'] = l1['EF']-(l1['EP'] if np.any(l1['EP']<=2) else 2+0.1*(l1['EP']-2) if np.any(l1['EP']>2) and np.any(l1['EP']<=10) else 2.8+0.001*(l1['EP']-10))
    l1['Kul'] = l1['EF']/(l1['NF']+l1['EP'])
    l1['Barinel'] = l1['EF']/(l1['EP']+l1['EF'])
	...
	...
	...
	...
	etc
    
    
    l1 = l1.replace([np.inf, -np.inf], np.nan)
    l1 = l1.fillna(0)
    YY = l1.target
    
    final_susp = l1['ER1a'] #The formula we are using to compute suspiciousness scores


   
    print(j, final_susp)

    sq=final_susp.tolist()

    path=file_list[j]
    with open(path,'w') as file_object:
        json.dump(sq,file_object)


# Wasted Effort Computation
fit1=[]
expense=[]
fault_position = [[976], [1116],........[379]]
total_statement = [3393, 3393,..........1499]
dfi = [susp1, susp2, ........... susp25]
for version in range(len(dfi)):
    examlist = []
    susp = dfi[version]
    sortedSusp_v = np.sort(susp)[::-1]
    
    for j in range(len(fault_position[version])):
        fau = fault_position[version][j]
        sf = susp[fau]
        tie = np.where(sortedSusp_v==sf)
        gr = np.where(sortedSusp_v > sf)
    
    Worst = len(gr[0]) 
    Best = (len(tie[0]))
    WE = (Worst+Best)/2
    faultP = WE
    examlist.append(faultP)
    for item in examlist:
        a.append(item)
        avgscore = np.mean(a)
print('WE = ', round(avgscore))
Wasted_Effort = (round(avgscore))
Wasted_Effort.to_csv("C:/My Thesis/Paper two modification/Third paper rewrite/check.csv")

# Here is the code for the correlation of the wasted effort computed by each metric

file = "C:/Research Projects/location_of_the_WE.csv"
df = pd.read_csv(file)
scaler = MinMaxScaler()
scale = scaler.fit(df)
scaled_WE = scale.transform(df)
scaled_to_df = pd.DataFrame(scaled_WE, columns = df.columns)
correlation_result = scaled_to_df.corr()
correlation_result.to_csv("C:/Research Projects/correlated_result.csv")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This code is used to perform the main experiment study
# This is to compare the performance of the combined metrics and the existing metrics
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

dfLr = [df1,df2,df3,df4,........]
file_list=[
'C:/My Thesis/Paper two modification/New Experiment/Linear/fault1.json',
..
..
..
'C:/My Thesis/Paper two modification/New Experiment/Linear/fault449.json',]

for j in range(len(file_list)):
    l1=dfLr[j]
    l1.columns=['EP', 'EF', 'NP', 'NF', 'target']
    
    
    def di(x, y):
        return 0 if np.any(y == 0) else x / y
    def gp_div(x,y):
        try:
            return x/y
        except ZeroDivisionError:
            return 0
    
	# List of Existing formulas
    l1['OP2'] = l1['EF'] - (l1['EP']/(l1['EP']+l1['NP']+1))
    l1['ER1a'] = l1.apply(lambda x: -1.000 if (x[1] < (x[1] + x[3])) else x[2], axis=1)
    l1['Tar'] = np.divide((np.divide(l1['EF'],(l1['EF']+l1['NF']))),(np.divide(l1['EF'],(l1['EF']+l1['NF']))+np.divide(l1['EP'],(l1['EP']+l1['NP']))))
    l1['Och1'] = np.divide(l1['EF'],(np.sqrt(l1['EF']+l1['NF'])*(l1['EF']+l1['EP'])))
    l1['Och2'] = gp_div((l1['EF']*l1['NP']),((l1['EF']+l1['EP'])*(l1['NP']+l1['NF'])*(l1['EF']+l1['NF'])*(l1['EP']+l1['NP'])))
    l1['Amp'] = abs((l1['EF']/(l1['EF']+l1['NF']))-(l1['EP']/(l1['EP']+l1['NP'])))
    l1['Jac'] = l1['EF']/(l1['EF']+l1['EP']+l1['NF'])
    l1['D2'] = l1['EF']**2/(l1['EP']+l1['NF'])
    l1['GP2'] = 2*(l1['EF']+np.sqrt(l1['NP'])+np.sqrt(l1['EP']))
    l1['GP3'] = np.sqrt(abs(l1['EF']**2-np.sqrt(l1['EP'])))
    l1['GP13'] = l1['EF']*(1+(1/(2*l1['EP']+l1['EF'])))
    l1['GP19'] = l1['EF']*(np.sqrt(abs(l1['EP']-l1['EF']+l1['NF']-l1['NP'])))     
    l1['Wong1'] = l1['EF']
    l1['Wong2'] = (l1['EF']-l1['EP'])
    l1['Wong3'] = l1['EF']-(l1['EP'] if np.any(l1['EP']<=2) else 2+0.1*(l1['EP']-2) if np.any(l1['EP']>2) and np.any(l1['EP']<=10) else 2.8+0.001*(l1['EP']-10))
    l1['Kul'] = l1['EF']/(l1['NF']+l1['EP'])
    l1['Barinel'] = l1['EF']/(l1['EP']+l1['EF'])
	...
	...
	...
	...
	etc
	
	l1 = l1.replace([np.inf, -np.inf], np.nan)
    l1 = l1.fillna(0)
	
	sc = MinMaxScaler()
    scale = sc.fit(l1)
    scaler = scale.transform(l1) # After each metric compute the suspiciousness, we scale the result to be able to combine them.
    
    scaled_susp = pd.DataFrame(scaler, columns = l1.columns)
	
	# Just to mention few of the formulas for demonstration. We updated all the formulas in this section
	A = scaled_susp["OP2"] 
    B = scaled_susp["ER1a"]
    C = scaled_susp["Och1"]
    D = scaled_susp["Och2"]
    E = scaled_susp["Tar"]
    F = scaled_susp["Amp"]
    G = scaled_susp["Jac"]
    H = scaled_susp["D2"]
    I = scaled_susp['Barinel']
    J = scaled_susp['Rogot1']
    K = scaled_susp['Tar']
    L = scaled_susp['Wong2']
	
	# We added another columns to the existing datafile to show update results for the combination
	# This is just an example. Consult our paper to check the main combined metrics
    scaled_susp['comb1'] = A + B
    scaled_susp['comb2'] = A + C
    scaled_susp['comb3'] = B + C
    scaled_susp['comb4'] = D + E
    scaled_susp['comb5'] = F + G
    scaled_susp['comb6'] = C + H
    scaled_susp['comb7'] = E + F
    scaled_susp['comb8'] = I + J
    scaled_susp['comb9'] = K + L
    
	
	final_susp = l12['comb9'] # This is the final suspiciousness score

    print(j, final_susp)

    sq=final_susp.tolist()

    path=file_list[j]
    with open(path,'w') as file_object:
        json.dump(sq,file_object)
		

# Opening of the dumped files for ranking
susp1	=json.load(open('E:/fault1.json'))
susp2	=json.load(open('E:/fault2.json'))
..
..
..
..
susp449	=json.load(open('E:/fault449.json'))



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Ranking of the suspiciousness score
# This code is Computing Exam Score, Wasted Effort, Mean Average Precision, and 
#Accuracy for the  main experiment used to compare the studied methods
# We run the same code for small, medium, and large datasets as partitioned in the paper
# The fault_position, total_statement, and dfi values are changed for each category
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
fit1=[]
expense=[]
fault_position = [[976], [1116],........[379]]
total_statement = [3393, 3393,..........1499]

dfi = [susp1, susp2, ........... susp(n)]

# Exam Score Computation
for version in range(len(dfi)):
    fit=[]
    R=[]
    key=[]
    tie = []
    gr = []
    a = []
    k = []
    for h in range(len(dfi[version])):
        R.append(h)
    susp=dict(zip(R,dfi[version]))
    susp_x=sorted(susp.items(),key = lambda x:x[1],reverse = True)
    for p in range(len(susp_x)):
        key.append(susp_x[p][0])
    for t in range(len(key)):
        for j in range(len(fault_position[version])):
            if key[t]==fault_position[version][j]:
                fit.append(t)
    fit1.append(fit[0])
    #WE = (len(tie)+len(gr))/2
    expense.append((fit[0]/total_statement[version])*100)
    

print(fit1)
print(expense)

# Mean Average Precision Computation   
for version in range(len(dfi)):
    count = 0
    found = 0
    prec_list = []
    R=[]
    t = []
    for h in range(len(dfi[version])):
        R.append(h)
    susp=dict(zip(R,dfi[version]))
    susp_x=sorted(susp.items(),key = lambda x:x[1],reverse = True)
    for p in susp_x:
        count+=1
        method_id = p[0]
        if method_id in fault_position[version]:
            found += 1
            precision = float(found) / float(count)
            prec_list += [precision]
            for c in prec_list:
                t.append(c)
    if len(prec_list) == 0:
        print("ERROR: no matches in average precisions")
        print(fault_position[version])
    q = np.mean(prec_list)
    k.append(q)
prec = np.mean(k)  #np.mean(t) 

    
# Wasted Effort Computation    
for version in range(len(dfi)):
    examlist = []
    susp = dfi[version]
    sortedSusp_v = np.sort(susp)[::-1]
    
    for j in range(len(fault_position[version])):
        fau = fault_position[version][j]
        sf = susp[fau]
        tie = np.where(sortedSusp_v==sf)
        gr = np.where(sortedSusp_v > sf)
    
    Worst = len(gr[0]) 
    Best = (len(tie[0]))
    WE = (Worst+Best)/2
    faultP = WE
    examlist.append(faultP)
    for item in examlist:
        a.append(item)
        avgscore = np.mean(a)

  
# Accuracy computation
acc1=0
acc3=0
acc5=0
acc10=0
for k in range(len(fit1)):
    if fit1[k] <= 1:
        acc1=acc1+1

for l in range(len(fit1)):
    if fit1[l] <= 3:
        acc3=acc3+1

for m in range(len(fit1)):
    if fit1[m] <= 5:
        acc5=acc5+1        

for n in range(len(fit1)):
    if fit1[n] <= 10:
        acc10=acc10+1

def stddev(data):
    mean = sum(data) / len(data)
    return np.sqrt((1/len(data)) * sum((i-mean)**2 for i in data))

print('acc1=',(acc1/len(stateS))*(100))
print('acc3=',(acc3/len(stateS))*(100))
print('acc5=',(acc5/len(stateS))*(100))
print('acc10=',(acc10/len(stateS))*(100))

print('expense=',round(np.mean(expense),3))
print('MAP = ', prec)
print('WE = ', round(avgscore))

print('STD=',stddev(data=a))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Reporting the final result in CSV file
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

small_acc1 = round((acc1/len(stateS))*(100))
small_acc3 = round((acc3/len(stateS))*(100))
small_acc5 = round((acc5/len(stateS))*(100))
small_acc10 = round((acc10/len(stateS))*(100))
small_expense = round(np.mean(expense),2)
small_MAP = round(prec,3)
small_Wasted_Effort = round(avgscore)
small_Standard_deviation = round(stddev(data=a),2)

medium_acc1 = round((acc1/len(stateS))*(100))
medium_acc3 = round((acc3/len(stateS))*(100))
medium_acc5 = round((acc5/len(stateS))*(100))
medium_acc10 = round((acc10/len(stateS))*(100))
medium_expense = round(np.mean(expense),2)
medium_MAP = round(prec,3)
medium_Wasted_Effort = round(avgscore)
medium_Standard_deviation = round(stddev(data=a),2)

large_acc1 = round((acc1/len(stateS))*(100))
large_acc3 = round((acc3/len(stateS))*(100))
large_acc5 = round((acc5/len(stateS))*(100))
large_acc10 = round((acc10/len(stateS))*(100))
large_expense = round(np.mean(expense),2)
large_MAP = round(prec,3)
large_Wasted_Effort = round(avgscore)
large_Standard_deviation = round(stddev(data=a),2)

print('Small programs')
print(small_acc1)
print(small_acc3)
print(small_acc5)
print(small_acc10)
print(small_expense)
print(small_MAP)
print(small_Wasted_Effort)
print(small_Standard_deviation)

print('Medium programs')
print(medium_acc1)
print(medium_acc3)
print(medium_acc5)
print(medium_acc10)
print(medium_expense)
print(medium_MAP)
print(medium_Wasted_Effort)
print(medium_Standard_deviation)

print('Large programs')
print(large_acc1)
print(large_acc3)
print(large_acc5)
print(large_acc10)
print(large_expense)
print(large_MAP)
print(large_Wasted_Effort)
print(large_Standard_deviation)

small_progs = list([small_acc1, small_acc3, small_acc5, small_acc10, small_expense, small_MAP, small_Wasted_Effort, small_Standard_deviation])
medium_progs = list([medium_acc1, medium_acc3, medium_acc5, medium_acc10, medium_expense, medium_MAP, medium_Wasted_Effort, medium_Standard_deviation])
large_progs = list([large_acc1, large_acc3, large_acc5, large_acc10, large_expense, large_MAP, large_Wasted_Effort, large_Standard_deviation])

output_dataframe = pd.DataFrame([small_progs, medium_progs, large_progs])

output_dataframe.to_csv("C:/output_dataframe.csv")
