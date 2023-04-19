import csv
import datetime

import pandas as pd
import yfinance
import os
import pandas
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import linregress
import numpy as np


def getdatacsv():
    mainDf = pandas.read_csv(os.getcwd() + '\\MainData\\TickerData.csv', encoding='latin1', header=0)
    return mainDf


def removecols(mainDf):
    if 'Transfer' in mainDf.columns:
        mainDf.drop(['Transfer'], axis=1,inplace = True)
    if 'LSE IPO' in mainDf.columns:
        mainDf.drop(['LSE IPO'], axis=1,inplace = True)
    if 'notes' in mainDf.columns:
        mainDf.drop(['notes'], axis=1,inplace = True)
    return mainDf


def checkfortypes(mainDf):
    print(mainDf.isna().sum(),"\n")


def fixemptydata(mainDf):
    mainDf['FTSE Industry'].fillna(0,inplace = True)
    mainDf['FTSE Sector'].fillna('Not Identified',inplace = True)
    mainDf[' FTSE Subsector '].fillna('Not Identified',inplace = True) #not sure why there is spaces
    mainDf['Currency'].fillna('GBX',inplace = True)
    mainDf['Nominated Advisor (AIM only)'].fillna('No Advisor',inplace = True)
    return mainDf

def showrowswithmissingvals(mainDf):
    values_missing = ["n.a.", "?", "NA", "n/a", "na", "--"]
    missingValCol = (mainDf.isnull().sum())
    print(missingValCol[missingValCol > 0])

def removeemptyrows(mainDf):
    #drop row if column TIDM is NAN
    mainDf.dropna(subset=['TIDM'],inplace = True)
    mainDf.dropna(subset=['Initial Trading Open'],inplace = True)
    return mainDf

def rowcount(mainDf):
    print("Dataframe has", len(mainDf.index),"rows")


def removeanomalousrows(mainDf):
    #print(mainDf['Initial Trading Open'].dtype)
    tickersToRemove = ['SENX','MTPH','MYSQ','BARK','CRTM','FISH','IL0A','PRSM']
    for ticker in tickersToRemove:
        mainDf.drop(index=mainDf[mainDf['TIDM'] == ticker].index,inplace = True)

    #Yahoo picking up old data for old company tickers/Delisted-
    #SENX
    #MTPH
    #MYSQ
    #BARK
    #Fish
    #Random jumps in data
    #CRTM
    #IL0A
    #PRSM
    return mainDf

def cleancolumns(mainDf):
    mainDf = cleanissueprice(mainDf)
    mainDf = cleanmarketcap(mainDf)
    mainDf = cleanmoneyraisedexisting(mainDf)
    mainDf = cleantotalraised(mainDf)
    return mainDf


def cleanissueprice(mainDf):
    cleanDf = mainDf.dropna(subset=['Issue Price'])
    slope, intercept, rvalue, pvalue, stderr = linregress(cleanDf['Issue Price'], cleanDf['Adj Close Day 1'])
    x = cleanDf['Issue Price']
    y = slope * x + intercept
    mainDf['Issue Price'] = mainDf.apply(
        lambda row: row['Adj Close Day 1'] / slope - intercept if np.isnan(row['Issue Price']) else row['Issue Price'],
        axis=1)
    return mainDf


def cleanmarketcap(mainDf):
    mainDf.loc[mainDf[' Market Cap - Opening Price (Â£m) '] == ' -   ', ' Market Cap - Opening Price (Â£m) '] = 0
    return mainDf

def cleanmoneyraisedexisting(mainDf):
    mainDf.loc[mainDf['Money Raised - Existing ()'] == ' -   ', 'Money Raised - Existing ()'] = 0
    return mainDf

def cleantotalraised(mainDf):
    mainDf.loc[mainDf['Money Raised - Existing ()'] == ' -   ', 'Money Raised - Existing ()'] = 23
    return mainDf

def encodingcatagorical(mainDf):

    mainDf['Market'] = mainDf['Market'].astype('category')
    mainDf['Market_new'] = mainDf['Market'].cat.codes
    encoder = OneHotEncoder(handle_unknown='error')
    encData = pd.DataFrame(encoder.fit_transform(mainDf[['Market_new']]).toarray())
    mainDf = mainDf.join(encData)
    #print(mainDf)
    return mainDf

def datapreparation(mainDf):
    # use showrowswithmissingvals(mainDf)
    mainDf = removeemptyrows(mainDf)
    mainDf = removeanomalousrows(mainDf)
    mainDf = removecols(mainDf)
    mainDf = fixemptydata(mainDf)
    mainDf = cleancolumns(mainDf)
    mainDf = encodingcatagorical(mainDf)
    #checkfortypes(mainDf) #all rows shown
    #rowcount(mainDf) #how many rows in df
    mainDf.to_csv(os.getcwd() + '\\MainData\\DataPrep.csv', index=False)  # save for later viewing
    for col in mainDf:
        if "Adj Close Day" not in col:
            mainDf.dropna(subset=[col], inplace=True)
    showrowswithmissingvals(mainDf)
    return mainDf




def issueprice(mainDf):
    cleanDf = mainDf.dropna(subset=['Issue Price'])
    plt.scatter(cleanDf['Issue Price'], cleanDf['Adj Close Day 1'])
    plt.xlabel('Issue Price')
    plt.ylabel('Adj Close Day 1')
    slope, intercept, rvalue, pvalue, stderr = linregress(cleanDf['Issue Price'], cleanDf['Adj Close Day 1'])
    print('Slope:', slope)
    print('Intercept:', intercept)
    print('R-squared:', rvalue ** 2)
    x = cleanDf['Issue Price']
    y = slope * x + intercept
    plt.plot(x, y, color='r')
    plt.show()
    mainDf['Issue Price'] = mainDf.apply(lambda row: row['Adj Close Day 1'] / slope - intercept if np.isnan(row['Issue Price']) else row['Issue Price'], axis=1)


def moneyraised(mainDf):
    cleanDf = mainDf.drop(mainDf[mainDf['Money Raised - Existing ()'] == '-'].index)
    cleanDf['Money Raised - Existing ()'] = cleanDf['Money Raised - Existing ()'].astype(float)
    mean = cleanDf['Money Raised - Existing ()'].mean()
    median = cleanDf['Money Raised - Existing ()'].median()
    print("Money Raised - Existing ()", mean)
    print("Money Raised - Existing ()", median)
    plt.hist(cleanDf['Money Raised - Existing ()'], bins=70)
    plt.axvline(x=mean, color='r', linestyle='--')     # Add a vertical line for the mean
    plt.axvline(x=median, color='#FFA500', linestyle='--')  # Add a vertical line for the mean
    plt.xlabel('Money Raised - Existing ()')
    plt.ylabel('Frequency')
    plt.title('Histogram of Money Raised - Existing ()')
    plt.show()

    cleanDf['Money Raised - Existing ()'].plot(kind='density')
    plt.xlabel('Money Raised - Existing ()')
    plt.ylabel('Density')
    #plt.grid(True)
    plt.show()


def totalraised(mainDf):
    cleanDf = mainDf.drop(mainDf[mainDf['TOTAL RAISED ()'] == '-'].index)
    cleanDf['TOTAL RAISED ()'] = cleanDf['TOTAL RAISED ()'].astype(float)
    mean = cleanDf['TOTAL RAISED ()'].mean()
    median = cleanDf['TOTAL RAISED ()'].median()
    print("Mean of TOTAL RAISED ():", mean)
    print("Median of TOTAL RAISED ():",median)
    plt.hist(cleanDf['TOTAL RAISED ()'], bins=70)
    plt.axvline(x=mean, color='r', linestyle='--')  # Add a vertical line for the mean
    plt.axvline(x=median, color='#FFA500', linestyle='--')  # Add a vertical line for the mean
    plt.xlabel('TOTAL RAISED ()')
    plt.ylabel('Frequency')
    plt.title('Histogram of TOTAL RAISED ()')
    plt.show()

    cleanDf['TOTAL RAISED ()'].plot(kind='density')
    plt.xlabel('TOTAL RAISED ()')
    plt.ylabel('Density')
    # plt.grid(True)
    plt.show()

def topfeatures(mainDf):
    k = '5'
    initialDayDf = mainDf.iloc[:, 0:22]
    for col in initialDayDf:
        # Skip columns that are not numeric
        if not pandas.api.types.is_numeric_dtype(initialDayDf[col]):
            continue

        # Create a SelectKBest object to select the top k features for the column
        selector = SelectKBest(score_func=f_classif, k=k)

        top_10_features = SelectKBest(chi2, k=10).fit_transform(initialDayDf, initialDayDf['Adj Close Day 1'])
        print(top_10_features)


def exploratorydataanalysis(mainDf):
    issueprice(mainDf)
    moneyraised(mainDf)
    totalraised(mainDf)
    topfeatures(mainDf)
    #mainDf['Currency'].value_counts().plot.bar()
    #matplotlib.pyplot.show()

    # mainDf['Initial Trading Open'].plot.bar()
    #matplotlib.pyplot.boxplot(mainDf['Initial Trading Open'], showmeans=True)
    # print(mainDf['Initial Trading Open'].max())
    # matplotlib.pyplot.figure()
    #matplotlib.pyplot.show()
    return mainDf

def main():
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.width', 280)

    mainDf = getdatacsv()
    mainDf = datapreparation(mainDf)
    mainDf = exploratorydataanalysis(mainDf)







main()