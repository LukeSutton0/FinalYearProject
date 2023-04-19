import csv
import datetime
import os
import pandas
import pandas as pd
import yfinance
# Read the CSV file into a list of dictionaries

# def openFileToRead():
#     tickerList = []
#     directory = os.getcwd() + '\\MainData'
#     for filename in os.listdir(directory):
#         # Check if the file name starts with "New issues"
#         if filename.startswith("New issues"):
#             with open(filename, 'r') as file:
#                 reader = csv.reader(file)
#                 for row in reader:
#                     if row[5] == 'TIDM':
#                         pass
#                     else:
#                         tickerList.append(row[5] + '.L')
#     return tickerList

# def adjustDayList(dayList):
#     dayListAlt = []
#     for i in range(len(dayList)):
#         if i == 0:
#             dayListAlt.append(dayList[i])
#         else:
#             dayListAlt.append(dayList[i] - dayList[i - 1])
#     return dayListAlt

def endDateCalculate(datetimeObj,dayList):
    datesToPrint = []
    endDates = []
    for date in dayList:
        endDate = datetimeObj + datetime.timedelta(days=date)
        endDates.append(endDate)
    for endDate in endDates:

        datesToPrint.append(endDate.strftime('%Y-%m-%d'))
    return datesToPrint


def adjCloseCalc(mainDf,index,dataFromYahoo,daysToAdd):
    date = dataFromYahoo.index.min()
    try:
        date = date + pd.Timedelta(days=daysToAdd-1)
        date_str = date.strftime("%Y-%m-%d")
        currentDate = datetime.datetime.now().strftime("%Y-%m-%d")
        if date_str > currentDate:
            pass
        else:
            column = 'Adj Close Day '+str(daysToAdd)
            if date in dataFromYahoo.index:
                mainDf.at[index.Index, column] = dataFromYahoo.loc[date]['Adj Close']
            else:
                count = 0
                for aDate in dataFromYahoo.index:
                    if aDate > date:
                        count2 = 0
                        for row in dataFromYahoo.index:
                            if count-1 == count2:
                                mainDf.at[index.Index, column] = dataFromYahoo.loc[row]['Adj Close']
                                break
                            count2+=1
                        mainDf.at[index.Index, column] = dataFromYahoo.loc[aDate]['Adj Close']
                        break
                    count+=1
    except:
        print("error converting Adj close")

    return mainDf



def main():
        mainDf = pandas.DataFrame()
        try:
            directory = os.getcwd() + '\\MainData'
            for filename in os.listdir(directory):
                     if filename.startswith("New issues"):
                        #print(filename)
                        mainDf = pandas.read_excel(directory+'\\'+filename,header=7)
        except:
            print("Data file not found")

        if mainDf.empty == False:
            for index in mainDf.itertuples():
                try:
                   if index.TIDM != "": #if ticker exists
                        dataFromYahoo = yfinance.download(index.TIDM+".L", period="max", repair=True, interval="1d",progress=False)
                        if dataFromYahoo.empty:
                            pass
                        else:
                            mainDf.at[index.Index, 'Initial Trading Open'] = dataFromYahoo.loc[dataFromYahoo.index.min()]['Open']
                            mainDf.at[index.Index, 'Initial Trading High'] = dataFromYahoo.loc[dataFromYahoo.index.min()]['High']
                            mainDf.at[index.Index, 'Initial Trading Low'] = dataFromYahoo.loc[dataFromYahoo.index.min()]['Low']
                            mainDf.at[index.Index, 'Adj Close Day 1'] = dataFromYahoo.loc[dataFromYahoo.index.min()]['Adj Close']
                            mainDf.at[index.Index, 'Volume Initial Trading Day'] = dataFromYahoo.loc[dataFromYahoo.index.min()]['Volume']
                            daysToAddList = [7,30,90,180,365,730,1825,3650]
                            for day in daysToAddList:
                                mainDf = adjCloseCalc(mainDf,index,dataFromYahoo,day)
                except:
                    #error with download
                    pass
            mainDf.to_csv(os.getcwd()+'\\MainData\\TickerData.csv',index=False) #save for later viewing


main()

#graveyard
#pandas.set_option('display.max_columns', None)
# df = pd.read_excel(os.getcwd() + '\\MainCsvs\\New issues and IPOs_37.xlsx', sheet_name='New issues and IPOs_37',header=7)
# df.to_csv(os.getcwd() + '\\MainCsvs\\New issues and IPOs.csv', index=False)