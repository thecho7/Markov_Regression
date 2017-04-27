#########################################################
# This code is for calculation of markov_autoregression
# 2017/01/16 S.H. Cho
# Email : thecho7@gist.ac.kr
#########################################################


#%matplotlib inline

####### Import dependencies #####
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sn
from decimal import Decimal
# NBER recessions
from pandas_datareader.data import DataReader
from datetime import datetime
usrec = DataReader('USREC', 'fred', start=datetime(1947, 1, 1), end=datetime(2013, 4, 1))
import csv


######## Load data #####

temp = open('1_r.csv', 'r') # 1_r.csv is a file with raw dataset
temp2 = open('result1.csv', 'a') # result1.csv is a resulting file will be created.
temp3 = open('2_r.csv', 'r') # 1_r.csv is a file with raw dataset

raw = csv.reader(temp)
raw2 = csv.reader(temp3)


####### Calculate the number of companies #####

# matrix size is the same as the number of company

company = 1004 # First index in the list of companies (1004)
compidx = company
i = 0
yr = 0
comp_yr_match = []
compid = [] # Store the companys' ID in an array
compyr = [] # Store the companys' year series in an array
row = 0

compid.append('1004')

for row in raw:
    compyr.append(row[1])
    if row[0] == str(compidx):
        yr = yr + 1
        continue
    else:
        comp_yr_match.append(yr)
        i = i + 1
        compidx = row[0]
        compid.append(row[0])
        yr = 1

comp_yr_match.append(yr)
i = i + 1

#print(compyr)
#print(comp_yr_match)

gnpmat = np.zeros((i, 21)) # gnpmat is a matrix with size i by 21. i is the number of companies and 21 is the maximum length of the annual data.

j = 0
k = 0

####### Append the data in the matrix (gnpmat). In every case, row[14] indicates 'c_compen'

for row in raw2:
    if row[0] == str(company):
        if row[14]:
            gnpmat[k][j] = row[14]
            j = j + 1
        else:
            continue
    else:
        company = row[0]
        j = 0
        k = k + 1

i=0
j=0
k=0
l=0

#print(len(gnpmat))
#print(len(gnpmat[1,:]))
#print(len(comp_yr_match))

####### By utilizing the data in gnpmat, calculate filtered marginal probabilities. After calculation, complete the file we want.
####### Consider only the data has length over 10. The resulting csv file has float numbers, inf (infinite), NaN (Not a number) and 0.
####### If you want, you can use res_hamilton.smoothed_marginal_probabilities[0] instead of res_hamilton.filtered_marginal_probabilities[0]
####### Clearly, you can utilize the result with (res_hamilton.filtered_marginal_probabilities[0], 1 - res_hamilton.filtered_marginal_probabilities[0])
####### res_hamilton.filtered_marginal_probabilities[0] + (1 - res_hamilton.filtered_marginal_probabilities[0]) = 1

for i in range(0, len(gnpmat)):
        compgnp = gnpmat[i]
        print i
        final = compgnp[~(compgnp==0)]
        print final
        if len(final) == 0:
            #dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1952-01-01', freq='QS'))
            #mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            #res_hamilton = mod_hamilton.fit()
            #print("FILTERED_MARGINAL_PROBABILITY\n")
            #print(res_hamilton.filtered_marginal_probabilities[0])
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            continue
        elif len(final) == 1:
            #dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1952-04-01', freq='QS'))
            #mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            #res_hamilton = mod_hamilton.fit()
            #print("FILTERED_MARGINAL_PROBABILITY\n")
            #print(res_hamilton.filtered_marginal_probabilities[0])
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            continue
        elif len(final) == 2:
            #dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1952-07-01', freq='QS'))
            #mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            #res_hamilton = mod_hamilton.fit()
            #print("FILTERED_MARGINAL_PROBABILITY\n")
            #print(res_hamilton.filtered_marginal_probabilities[0])
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            continue
        elif len(final) == 3:
            #dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1951-10-01', freq='QS'))
            #mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            #res_hamilton = mod_hamilton.fit()
            #print("FILTERED_MARGINAL_PROBABILITY\n")
            #print(res_hamilton.filtered_marginal_probabilities[0])
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            continue
        elif len(final) == 4:
            #dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1952-01-01', freq='QS'))
            #mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            #res_hamilton = mod_hamilton.fit()
            #print("FILTERED_MARGINAL_PROBABILITY\n")
            #print(res_hamilton.filtered_marginal_probabilities[0])
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            continue
        elif len(final) == 5:
            #dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1952-04-01', freq='QS'))
            #mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            #res_hamilton = mod_hamilton.fit()
            #print("FILTERED_MARGINAL_PROBABILITY\n")
            #print(res_hamilton.filtered_marginal_probabilities[0])
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            continue
        elif len(final) == 6:
            #dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1952-07-01', freq='QS'))
            #mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            #res_hamilton = mod_hamilton.fit()
            #print("FILTERED_MARGINAL_PROBABILITY\n")
            #print(res_hamilton.filtered_marginal_probabilities[0])
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            continue
        elif len(final) == 7:
            #dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1952-10-01', freq='QS'))
            #mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            #res_hamilton = mod_hamilton.fit()
            #print("FILTERED_MARGINAL_PROBABILITY\n")
            #print(res_hamilton.filtered_marginal_probabilities[0])
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            continue
        elif len(final) == 8:
            #dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1953-01-01', freq='QS'))
            #mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            #res_hamilton = mod_hamilton.fit()
            #print("FILTERED_MARGINAL_PROBABILITY\n")
            #print(res_hamilton.filtered_marginal_probabilities[0])
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            continue
        elif len(final) == 9:
            #dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1953-04-01', freq='QS'))
            #mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            #res_hamilton = mod_hamilton.fit()
            #print("FILTERED_MARGINAL_PROBABILITY\n")
            #print(res_hamilton.filtered_marginal_probabilities[0])
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            continue
        elif len(final) == 10:
            #dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1953-07-01', freq='QS'))
            #mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            #res_hamilton = mod_hamilton.fit()
            #print("FILTERED_MARGINAL_PROBABILITY\n")
            #print(res_hamilton.filtered_marginal_probabilities[0])
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            print('\n')
            continue
        elif len(final) == 11:
            dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1953-10-01', freq='QS'))
            mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            res_hamilton = mod_hamilton.fit()
            print("FILTERED_MARGINAL_PROBABILITY\n")
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            temp2.write('\t')
            for j in range(0, len(res_hamilton.filtered_marginal_probabilities[0])):
                temp2.write(str(res_hamilton.filtered_marginal_probabilities[0][j]) + '\t')
            temp2.write('\n')
        elif len(final) == 12:
            dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1954-01-01', freq='QS'))
            mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            res_hamilton = mod_hamilton.fit()
            print("FILTERED_MARGINAL_PROBABILITY\n")
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            temp2.write('\t')
            for j in range(0, len(res_hamilton.filtered_marginal_probabilities[0])):
                temp2.write(str(res_hamilton.filtered_marginal_probabilities[0][j]) + '\t')
            temp2.write('\n')
        elif len(final) == 13:
            dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1954-04-01', freq='QS'))
            mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            res_hamilton = mod_hamilton.fit()
            print("FILTERED_MARGINAL_PROBABILITY\n")
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            temp2.write('\t')
            for j in range(0, len(res_hamilton.filtered_marginal_probabilities[0])):
                temp2.write(str(res_hamilton.filtered_marginal_probabilities[0][j]) + '\t')
            temp2.write('\n')
        elif len(final) == 14:
            dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1954-07-01', freq='QS'))
            mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            res_hamilton = mod_hamilton.fit()
            print("FILTERED_MARGINAL_PROBABILITY\n")
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            temp2.write('\t')
            for j in range(0, len(res_hamilton.filtered_marginal_probabilities[0])):
                temp2.write(str(res_hamilton.filtered_marginal_probabilities[0][j]) + '\t')
            temp2.write('\n')
        elif len(final) == 15:
            dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1954-10-01', freq='QS'))
            mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            res_hamilton = mod_hamilton.fit()
            print("FILTERED_MARGINAL_PROBABILITY\n")
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            temp2.write('\t')
            for j in range(0, len(res_hamilton.filtered_marginal_probabilities[0])):
                temp2.write(str(res_hamilton.filtered_marginal_probabilities[0][j]) + '\t')
            temp2.write('\n')
        elif len(final) == 16:
            dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1955-01-01', freq='QS'))
            mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            res_hamilton = mod_hamilton.fit()
            print("FILTERED_MARGINAL_PROBABILITY\n")
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            temp2.write('\t')
            for j in range(0, len(res_hamilton.filtered_marginal_probabilities[0])):
                temp2.write(str(res_hamilton.filtered_marginal_probabilities[0][j]) + '\t')
            temp2.write('\n')
        elif len(final) == 17:
            dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1955-04-01', freq='QS'))
            mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            res_hamilton = mod_hamilton.fit()
            print("FILTERED_MARGINAL_PROBABILITY\n")
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            temp2.write('\t')
            for j in range(0, len(res_hamilton.filtered_marginal_probabilities[0])):
                temp2.write(str(res_hamilton.filtered_marginal_probabilities[0][j]) + '\t')
            temp2.write('\n')
        elif len(final) == 18:
            dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1955-07-01', freq='QS'))
            mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            res_hamilton = mod_hamilton.fit()
            print("FILTERED_MARGINAL_PROBABILITY\n")
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            temp2.write('\t')
            for j in range(0, len(res_hamilton.filtered_marginal_probabilities[0])):
                temp2.write(str(res_hamilton.filtered_marginal_probabilities[0][j]) + '\t')
            temp2.write('\n')
        elif len(final) == 19:
            dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1955-10-01', freq='QS'))
            mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            res_hamilton = mod_hamilton.fit()
            print("FILTERED_MARGINAL_PROBABILITY\n")
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            temp2.write('\t')
            for j in range(0, len(res_hamilton.filtered_marginal_probabilities[0])):
                temp2.write(str(res_hamilton.filtered_marginal_probabilities[0][j]) + '\t')
            temp2.write('\n')
        elif len(final) == 20:
            dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1956-01-01', freq='QS'))
            mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            res_hamilton = mod_hamilton.fit()
            print("FILTERED_MARGINAL_PROBABILITY\n")
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            temp2.write('\t')
            for j in range(0, len(res_hamilton.filtered_marginal_probabilities[0])):
                temp2.write(str(res_hamilton.filtered_marginal_probabilities[0][j]) + '\t')
            temp2.write('\n')
        elif len(final) == 21:
            dta_hamilton = pd.Series(final, index=pd.date_range('1951-04-01', '1956-04-01', freq='QS'))
            mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
            res_hamilton = mod_hamilton.fit()
            print("FILTERED_MARGINAL_PROBABILITY\n")
            temp2.write(str(compid[i]) + '\t')
            for k in range(0, comp_yr_match[i]):
                temp2.write(str(compyr[l]) + '\t')
                l = l + 1
            temp2.write('\n')
            temp2.write('\t')
            for j in range(0, len(res_hamilton.filtered_marginal_probabilities[0])):
                temp2.write(str(res_hamilton.filtered_marginal_probabilities[0][j]) + '\t')
            temp2.write('\n')
        else:
            continue
        
#dta_hamilton = pd.Series(rgnp, index=pd.date_range('1951-04-01', '1984-10-01', freq='QS'))

# Plot the data
#dta_hamilton.plot(title='Growth rate of Real GNP', figsize=(12,3))

# Fit the model
#mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
#res_hamilton = mod_hamilton.fit()

#res_hamilton.summary()

#fig, axes = plt.subplots(2, figsize=(7,7))
#ax = axes[0]
#ax.plot(res_hamilton.filtered_marginal_probabilities[0])
#ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='k', alpha=0.1)
#ax.set_xlim(dta_hamilton.index[4], dta_hamilton.index[-1])
#ax.set(title='Filtered probability of recession')

#ax = axes[1]
#ax.plot(res_hamilton.smoothed_marginal_probabilities[0])
#ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='k', alpha=0.1)
#ax.set_xlim(dta_hamilton.index[4], dta_hamilton.index[-1])
#ax.set(title='Smoothed probability of recession')

#fig.tight_layout()

#print(res_hamilton.expected_durations)

temp.close()
temp2.close()
