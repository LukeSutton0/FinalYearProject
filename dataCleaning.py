import csv
import datetime
import os
import pandas
import yfinance
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
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


def datapreparation(mainDf):
    # use showrowswithmissingvals(mainDf)
    mainDf = removeemptyrows(mainDf)
    mainDf = removeanomalousrows(mainDf)
    mainDf = removecols(mainDf)
    mainDf = fixemptydata(mainDf)

    mainDf = exploratorydataanalysis(mainDf)


    #checkfortypes(mainDf) #all rows shown
    #rowcount(mainDf) #how many rows in df
    #mainDf.to_csv(os.getcwd() + '\\MainData\\DataPrep.csv', index=False)  # save for later viewing
    showrowswithmissingvals(mainDf)

    return mainDf


def exploratorydataanalysis(mainDf):

    issueprice(mainDf)
    #mainDf['Currency'].value_counts().plot.bar()
    #matplotlib.pyplot.show()

    # mainDf['Initial Trading Open'].plot.bar()
    #matplotlib.pyplot.boxplot(mainDf['Initial Trading Open'], showmeans=True)
    # print(mainDf['Initial Trading Open'].max())
    # matplotlib.pyplot.figure()
    #matplotlib.pyplot.show()

    #topfeatures(mainDf)
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


def topfeatures(mainDf):
    k = 1
    for col in mainDf:
        # Skip columns that are not numeric
        if not pandas.api.types.is_numeric_dtype(mainDf[col]):
            continue
        print(mainDf[col])
        # Create a SelectKBest object to select the top k features for the column
        selector = SelectKBest(score_func=f_classif, k=k)

        # Fit the selector to the data in the column
        selector.fit(mainDf[[col]], mainDf['Adj Close Day 1'])

        # Get the indices of the top k features for the column
        top_k_indices = selector.get_support(indices=True)

        # Get the names of the top k features for the column
        top_k_features = mainDf[[col]].columns[top_k_indices]

        # Print the results
        print(f"Top {k} features for column '{col}': {list(top_k_features)}")




def main():
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.width', 280)

    mainDf = getdatacsv()
    mainDf = datapreparation(mainDf)








main()