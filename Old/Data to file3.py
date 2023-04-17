import csv
import datetime
import os

import pandas
import yfinance
# Read the CSV file into a list of dictionaries



def openFileToRead():
    tickerList = []
    current_dir = os.getcwd()
    with open(current_dir+'\\MainCsvFolder\\New issues and IPOs_37.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[5] == 'TIDM':
                pass
            else:
                tickerList.append(row[5] + '.L')


    return tickerList



# Example usage
def adjustDayList(dayList):
    dayListAlt = []
    for i in range(len(dayList)):
        if i == 0:
            dayListAlt.append(dayList[i])
        else:
            dayListAlt.append(dayList[i] - dayList[i - 1])
    return dayListAlt

def endDateCalculate(datetimeObj,dayList):
    datesToPrint = []
    endDates = []
    for date in dayList:
        endDate = datetimeObj + datetime.timedelta(days=date)
        endDates.append(endDate)
    for endDate in endDates:

        datesToPrint.append(endDate.strftime('%Y-%m-%d'))
    return datesToPrint

def main():
        pandas.set_option('display.max_columns', None)
        tickerList = openFileToRead()
        print(tickerList)

        for ticker in tickerList:
            print(ticker)
            data = yfinance.download(ticker, period="max",repair= True,interval = "1d")
            print(data.loc[data.index.min()])

            with open(os.getcwd() + '\\MainCsvFolder\\New issues and IPOs_37 - Copy.csv', 'r+') as file:
                reader = csv.reader(file)
                count = 0
                for row in reader:
                    if row[5] == ticker[:-2]:

                        writer = csv.writer(file)
                        row[19] = data.loc[data.index.min()]['Open']
                        row[20] = data.loc[data.index.min()]['High']
                        row[21] = data.loc[data.index.min()]['Low']
                        row[22] = data.loc[data.index.min()]['Adj Close']
                        row[23] = data.loc[data.index.min()]['Volume']

                        writer.writerows(reader)
                        file.flush()
                        break
                    else:
                        count +=1
                        pass


            #data.to_csv('New issues and IPOs_37 - Testing.csv', header=True, mode='a')

            print(data.index.min())



        myTicker = yfinance.Ticker(ticker)
        myTickerHist = myTicker.history(period="max")

        print(data.head(5))

        #print(myticker.cashflow)
        #print(f"Stock data for {myticker}:")
        #print(stock_data)




main()