import csv
import datetime
import os
import pandas
import yfinance
import matplotlib


def getdatacsv():
    mainDf = pandas.read_csv(os.getcwd() + '\\MainCsvs\\output.csv', encoding='latin1', header=0)
    return mainDf


def removecols(mainDf):
    mainDf = mainDf.drop(['Transfer', 'LSE IPO'], axis=1)
    return mainDf


def checkfortypes(mainDf):
    print(mainDf.isna().sum(),"\n")

def showrowswithmissingvals(mainDf):
    values_missing = ["n.a.", "?", "NA", "n/a", "na", "--"]
    missingValCol = (mainDf.isnull().sum())
    print(missingValCol[missingValCol > 0])

def removeemptyrows(mainDf):
    #drop row if column TIDM is NAN
    mainDf = mainDf.dropna(subset=['TIDM'])
    mainDf = mainDf.dropna(subset=['Initial Trading Open'])
    return mainDf

def rowcount(mainDf):
    print("Dataframe has", len(mainDf.index),"rows")


def removeanomalousrows(mainDf):
    #print(mainDf['Initial Trading Open'].dtype)
    tickersToRemove = ['SENX','MTPH','MYSQ','BARK','CRTM','FISH']
    for ticker in tickersToRemove:
        mainDf = mainDf.drop(index=mainDf[mainDf['TIDM'] == ticker].index)

    #Yahoo picking up old data for old company tickers/Delisted-
    #SENX
    #MTPH
    #MYSQ
    #BARK
    #Fish

    #Random jumps in data
    #CRTM
    return mainDf


def main():
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.width', 280)
    mainDf = getdatacsv()
    mainDf = removecols(mainDf)
    #showrowswithmissingvals(mainDf)
    mainDf = removeemptyrows(mainDf)
    mainDf = removeanomalousrows(mainDf)
    #showrowswithmissingvals(mainDf)
    #checkfortypes(mainDf)
    #rowcount(mainDf)

    mainDf['Currency'].value_counts().plot.bar()
    matplotlib.pyplot.show()
    #mainDf['Initial Trading Open'].plot.bar()
    matplotlib.pyplot.boxplot(mainDf['Initial Trading Open'],showmeans=True)
    #print(mainDf['Initial Trading Open'].max())
    #matplotlib.pyplot.figure()
    matplotlib.pyplot.show()
    #print(mainDf)






main()