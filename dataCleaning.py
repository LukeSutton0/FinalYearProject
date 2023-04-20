import csv
import datetime

import pandas as pd
import yfinance
import os
import pandas
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, chi2 , f_regression
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import linregress
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
    if 'TIDM' in mainDf.columns:
        mainDf.drop(['TIDM'], axis=1, inplace=True)
    if 'Company' in mainDf.columns:
        mainDf.drop(['Company'], axis=1, inplace=True)
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
    mainDf = cleandate(mainDf)
    mainDf = cleanmonth(mainDf)
    mainDf = cleanissueprice(mainDf)
    mainDf = cleanmarketcap(mainDf)
    mainDf = cleanmoneyraisedexisting(mainDf)
    mainDf = cleantotalraised(mainDf)
    mainDf = cleancountry(mainDf)
    return mainDf


def cleandate(mainDf):
    mainDf['Date'] = pd.to_datetime(mainDf['Date'])
    mainDf['Date'] = mainDf['Date'].dt.day
    mainDf.rename(columns={'Date': 'Day'}, inplace=True)
    #mainDf = mainDf.drop('Day',axis=1)
    return mainDf

def cleanmonth(mainDf):
    mainDf['Month'] = pd.to_datetime(mainDf['Month'])
    mainDf['Month'] = mainDf['Month'].dt.month
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
    mainDf.loc[(mainDf[' Market Cap - Opening Price (Â£m) '] == '-') | (mainDf[' Market Cap - Opening Price (Â£m) '] == " -   "), ' Market Cap - Opening Price (Â£m) '] = 0
    return mainDf

def cleanmoneyraisedexisting(mainDf):
    mainDf.loc[(mainDf['Money Raised - Existing ()'] == '-') | (mainDf['Money Raised - Existing ()'] == ' -   '), 'Money Raised - Existing ()'] = 0
    return mainDf

def cleantotalraised(mainDf):
    mainDf.loc[(mainDf['TOTAL RAISED ()'] == '-') | (mainDf['TOTAL RAISED ()'] == ' -   '), 'TOTAL RAISED ()'] = 23
    return mainDf

def cleancountry(mainDf):
    for index, row in mainDf.iterrows():
        if row['Country of Inc.'] != 'United Kingdom':
            mainDf.at[index, 'Country of Inc.'] = 'Not United Kingdom'
    return mainDf
def ftsecombine(mainDf):
    pandas.set_option('display.max_columns', None)
    catDf = mainDf.copy()
    if 'FTSE Sector' in catDf.columns:
        catDf.drop(['FTSE Sector'], axis=1, inplace=True)
    if ' FTSE Subsector ' in catDf.columns:
        catDf.drop([' FTSE Subsector '], axis=1, inplace=True)

    catDf['FTSE Industry'] = catDf['FTSE Industry'].astype(str)
    catDf.loc[catDf['FTSE Industry'] == 0,'FTSE Industry'] = "No Industry Data"
    # new system
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:2] == '10') & (catDf['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Technology"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:2] == '15') & (catDf['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Telecommunications"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:2] == '20') & (catDf['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Health Care"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:2] == '30') & (catDf['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Financials"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:2] == '35') & (catDf['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Real Estate"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:2] == '40') & (catDf['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Consumer Discretionary"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:2] == '45') & (catDf['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Consumer Staples"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:2] == '50') & (catDf['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Industrials"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:2] == '55') & (catDf['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Basic Materials"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:2] == '60') & (catDf['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Energy"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:2] == '65') & (catDf['FTSE Industry'].astype(str).str.len() > 6), 'FTSE Industry'] = "Utilities"

    #for old system
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:1] == '9') & (catDf['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Technology"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:1] == '6') & (catDf['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Telecommunications"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:1] == '4') & (catDf['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Health Care"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:1] == '8') & (catDf['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Financials"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:1] == '3') & (catDf['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Consumer Discretionary" #maybe look at this later
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:1] == '5') & (catDf['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Consumer Discretionary"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:1] == '2') & (catDf['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Industrials"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:1] == '1') & (catDf['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Basic Materials"
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:1] == '0') & (catDf['FTSE Industry'].astype(str).str.len() < 7) & (catDf['FTSE Industry'].astype(str).str.len() > 1), 'FTSE Industry'] = "Energy" #check this isnt interferring
    catDf.loc[(catDf['FTSE Industry'].astype(str).str[:1] == '7') & (catDf['FTSE Industry'].astype(str).str.len() < 7), 'FTSE Industry'] = "Utilities"

    return catDf


def encodingcatagorical(catDf):
    columnstocatagorise = ['Market','Issue type','Country of Inc.','FTSE Industry','Currency']
    for col in columnstocatagorise:
        #catDf['Market'] = catDf['Market'].astype('category') #change to category
        encoder = OneHotEncoder()
        encDf = pd.DataFrame(encoder.fit_transform(catDf[[col]]).toarray())
        colNames = encoder.get_feature_names_out([col])

        encDf.columns = colNames
        catDf.reset_index(drop=True,inplace=True)# reset index to ensure alignment
        catDf = pd.concat([catDf,encDf],axis=1)
        catDf = catDf.drop(col,axis=1)


    catDf.loc[catDf['Nominated Advisor (AIM only)'] != "No Advisor", 'Nominated Advisor (AIM only)'] = "Advisor"
    binarycoltocatagorise = ['Nominated Advisor (AIM only)']
    for col in binarycoltocatagorise:
        encoder = OneHotEncoder(drop='first')
        encDf = pd.DataFrame(encoder.fit_transform(catDf[[col]]).toarray())
        colNames = encoder.get_feature_names_out([col])
        encDf.columns = colNames
        catDf.reset_index(drop=True, inplace=True)  # reset index to ensure alignment
        catDf = pd.concat([catDf, encDf], axis=1)
        catDf = catDf.drop(col, axis=1)

    return catDf

def datapreparation(mainDf):
    # use showrowswithmissingvals(mainDf)
    # checkfortypes(mainDf) #all rows shown
    #rowcount(mainDf) #how many rows in df
    mainDf = removeemptyrows(mainDf)
    mainDf = removeanomalousrows(mainDf)
    mainDf = removecols(mainDf)
    mainDf = fixemptydata(mainDf)
    mainDf = cleancolumns(mainDf)
    for col in mainDf:
        if "Adj Close Day" not in col:
            mainDf.dropna(subset=[col], inplace=True)
    catDf = ftsecombine(mainDf)
    catDf = encodingcatagorical(catDf)
    catDf.to_csv(os.getcwd() + '\\MainData\\ColumnCheck.csv', index=False)  # save for later viewing
    return mainDf,catDf

def issuetype(mainDf):
    mode = mainDf['Issue type'].mode()
    mainDf['Issue type'].value_counts().plot.bar()
    plt.title("Types of issues placed onto the LSE")
    plt.ylabel("Quantity")
    plt.show()
    print(mode)

def countryofinc(mainDf):
    mode = mainDf['Country of Inc.'].mode()
    mainDf['Country of Inc.'].value_counts().plot.bar()
    plt.show()
    print(mode)
    counts = mainDf.groupby(mainDf['Country of Inc.'].eq('United Kingdom')).size()
    counts.plot.bar()
    plt.xticks([1, 0], ['UK','Non-Uk'], rotation=0)
    plt.xlabel('Country of Inc.')
    plt.ylabel('Count')
    plt.show()

def ftseindustry(mainDf):
    mainDf['FTSE Industry'].value_counts().plot.bar()
    plt.title("Bar chart showing quantity of IPOs per Industry sector")
    plt.ylabel("Quantity")
    plt.show()

def ftsesector(mainDf):
    print(mainDf)
    mainDf['FTSE Sector'].value_counts().plot.bar()
    plt.title('IPO quantity per sector')
    plt.xlabel('Quantity')
    plt.ylabel('FTSE Sector')
    plt.show()

    sns.stripplot(x="Year", y="FTSE Sector", data=mainDf, jitter=False)
    plt.xlabel('Year')
    plt.ylabel('FTSE Sector')
    plt.show()

    sector_counts = mainDf.groupby(['Year', 'FTSE Sector']).size().reset_index(name='counts')
    # Initialize empty list for plotted labels
    plotted_labels = []
    for sector in sector_counts['FTSE Sector'].unique():
        data = sector_counts[sector_counts['FTSE Sector'] == sector]
        if len(data) >= 3:  # exclude values with less than 5
            plt.plot(data['Year'], data['counts'], label=sector)
            plotted_labels.append(sector)  # add label to plotted labels list

    # Set legend to only include plotted labels
    plt.legend(loc='upper left', labels=plotted_labels, bbox_to_anchor=(1.05, 1))
    plt.title('IPOs released per sector over the last 20 years')

    plt.xlabel('Year')
    plt.xticks(rotation=0)
    plt.gca().xaxis.set_major_formatter('{:.0f}'.format)
    plt.ylabel('Quantity')
    plt.show()


def ftsesubsector(mainDf):
    mainDf[' FTSE Subsector '].value_counts().plot.bar()
    plt.title("Bar chart showing quantity of IPOs per Industry Subsector")
    plt.ylabel("Quantity")
    plt.show()

def issueprice(mainDf):
    tempDf = mainDf.dropna(subset=['Issue Price'])
    plt.scatter(tempDf['Issue Price'], tempDf['Adj Close Day 1'])
    plt.xlabel('Issue Price')
    plt.ylabel('Adj Close Day 1')
    slope, intercept, rvalue, pvalue, stderr = linregress(tempDf['Issue Price'], tempDf['Adj Close Day 1'])
    print('Slope:', slope)
    print('Intercept:', intercept)
    print('R-squared:', rvalue ** 2)
    x = tempDf['Issue Price']
    y = slope * x + intercept
    plt.plot(x, y, color='r')
    plt.show()
    mainDf['Issue Price'] = mainDf.apply(lambda row: row['Adj Close Day 1'] / slope - intercept if np.isnan(row['Issue Price']) else row['Issue Price'], axis=1)


def moneyraised(mainDf):
    tempDf = mainDf.drop(mainDf[mainDf['Money Raised - Existing ()'] == '-'].index)
    tempDf['Money Raised - Existing ()'] = tempDf['Money Raised - Existing ()'].astype(float)
    mean = tempDf['Money Raised - Existing ()'].mean()
    median = tempDf['Money Raised - Existing ()'].median()
    print("Money Raised - Existing ()", mean)
    print("Money Raised - Existing ()", median)
    plt.hist(tempDf['Money Raised - Existing ()'], bins=70)
    plt.axvline(x=mean, color='r', linestyle='--')     # Add a vertical line for the mean
    plt.axvline(x=median, color='#FFA500', linestyle='--')  # Add a vertical line for the mean
    plt.xlabel('Money Raised - Existing ()')
    plt.ylabel('Frequency')
    plt.title('Histogram of Money Raised - Existing ()')
    plt.show()

    tempDf['Money Raised - Existing ()'].plot(kind='density')
    plt.xlabel('Money Raised - Existing ()')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()


def totalraised(mainDf):
    tempDf = mainDf.drop(mainDf[mainDf['TOTAL RAISED ()'] == '-'].index)
    tempDf['TOTAL RAISED ()'] = tempDf['TOTAL RAISED ()'].astype(float)
    mean = tempDf['TOTAL RAISED ()'].mean()
    median = tempDf['TOTAL RAISED ()'].median()
    print("Mean of TOTAL RAISED ():", mean)
    print("Median of TOTAL RAISED ():",median)
    plt.hist(tempDf['TOTAL RAISED ()'], bins=70)
    plt.axvline(x=mean, color='r', linestyle='--')  # Add a vertical line for the mean
    plt.axvline(x=median, color='#FFA500', linestyle='--')  # Add a vertical line for the mean
    plt.xlabel('TOTAL RAISED £')
    plt.ylabel('Number of IPOs')
    plt.title('Histogram of total capital gained before sale')
    plt.show()

    tempDf['TOTAL RAISED ()'].plot(kind='density')
    plt.xlabel('TOTAL RAISED ()')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()


def day1topfeat(catDf):
    # day 1
    tempDf = catDf.copy()
    dropCols = list(tempDf.columns[9:21])
    tempDf = tempDf.drop(dropCols, axis=1)
    for col in tempDf:
        tempDf[col] = tempDf[col].astype(float)
    print(tempDf.shape)
    selector = SelectKBest(score_func=f_regression, k=7)
    topFeatures = selector.fit_transform(tempDf, catDf['Adj Close Day 1'])
    filter = selector.get_support()
    features = tempDf.columns[filter]
    featDf = pd.DataFrame(topFeatures)
    featDf.columns = features  # assign column names to featDf DataFrame
    print("All features:", features)
    corr = featDf.corr()
    plt.figure()
    plt.title("A heatmap showing the correlation between the variables used for linear regression")
    sns.heatmap(corr, annot=True, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.show()

    selector = SelectKBest(score_func=f_regression, k=7)
    topFeatures = selector.fit_transform(tempDf, catDf['Adj Close Day 1'])
    filter = selector.get_support()
    featScores = pd.DataFrame({'Feature': features, 'Fischer Score': selector.scores_[filter]})
    featScores = featScores.sort_values('Fischer Score', ascending=False)
    # Create a bar plot of the Fischer scores
    plt.bar(featScores['Feature'], featScores['Fischer Score'])
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Fischer Score')
    plt.title('Top 7 Fischer Scores')
    plt.show()


def topfeatures(catDf):
    day1topfeat(catDf)



def exploratorydataanalysis(mainDf,catDf):
    issuetype(mainDf)
    countryofinc(mainDf)
    ftseindustry(mainDf)
    ftsesector(mainDf)
    ftsesubsector(mainDf)
    issueprice(mainDf)
    moneyraised(mainDf)
    totalraised(mainDf)
    topfeatures(catDf)



    # plt.boxplot(mainDf['Initial Trading Open'], showmeans=True)
    # print(mainDf['Initial Trading Open'].max())
    # plt.title("Box plot showing Initial Trading Open price of stock")
    # plt.xlabel("Day 1")
    # plt.ylabel("Price (£)")
    # plt.figure()
    # plt.show()

    return mainDf


def predictday1(catDf):
    #drop data for later prediction
    dropCols = list(catDf.columns[9:21])
    catDf = catDf.drop(dropCols,axis=1)
    for col in catDf:
        catDf[col] = catDf[col].astype(float)
    #indicators from EDA : Issue Price', ' Market Cap - Opening Price (Â£m) ', 'Money Raised - Existing ()', 'TOTAL RAISED ()', 'Initial Trading Open', 'Market_International Main Market', 'FTSE Industry_Industrials'
    predictDf = catDf.copy()
    columnstokeep = ['Issue Price', ' Market Cap - Opening Price (Â£m) ', 'Money Raised - Existing ()', 'TOTAL RAISED ()', 'Initial Trading Open', 'Market_International Main Market', 'FTSE Industry_Industrials']
    for col in predictDf:
        if col not in columnstokeep:
            #drop col if not in columns to keep
            predictDf = predictDf.drop(col, axis=1)

    # pd.options.display.max_rows = None
    # pd.options.display.max_columns = None
    # print(catDf[catDf < 0].sum())
    # print((catDf < 0).any())



























def main():
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.width', 280)

    mainDf = getdatacsv()
    mainDf,catDf = datapreparation(mainDf)

    #mainDf = exploratorydataanalysis(mainDf,catDf)

    predictday1(catDf)






main()