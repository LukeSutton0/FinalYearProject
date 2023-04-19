import csv
import datetime
import os
import pandas
import yfinance
import matplotlib
from sklearn.feature_selection import SelectKBest, f_classif

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


def removenonipo(mainDf):
    rows_to_drop = mainDf[mainDf['LSE IPO'] == "Not IPO"].index
    mainDf.drop(rows_to_drop, inplace=True)
    return mainDf





def datapreparation(mainDf):
    mainDf = removecols(mainDf)
    mainDf = removenonipo(mainDf)
    # showrowswithmissingvals(mainDf)
    mainDf = removeemptyrows(mainDf)
    mainDf = removeanomalousrows(mainDf)
    showrowswithmissingvals(mainDf)
    # checkfortypes(mainDf)
    # rowcount(mainDf)
    #print(mainDf)


    mainDf = exploratorydataanalysis(mainDf)

    return mainDf


def exploratorydataanalysis(mainDf):

    mainDf['Currency'].value_counts().plot.bar()
    #matplotlib.pyplot.show()

    # mainDf['Initial Trading Open'].plot.bar()
    matplotlib.pyplot.boxplot(mainDf['Initial Trading Open'], showmeans=True)
    # print(mainDf['Initial Trading Open'].max())
    # matplotlib.pyplot.figure()
    #matplotlib.pyplot.show()

    topfeatures(mainDf)
    return mainDf


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