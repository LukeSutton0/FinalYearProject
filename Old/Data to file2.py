import csv
import datetime
import os
import shutil

# Read the CSV file into a list of dictionaries
filename = "ESO.L.csv"
mainFileName =  "New issues and IPOs_37.csv"
def openFileToRead():

    for i in os.listdir():
        if i.endswith('.csv'):
            print(i)
            with open(i, 'r') as file:
                reader = csv.DictReader(file)
                data = [row for row in reader]


    return data

def printDataOnDays(datesToPrint,data):
    for day in datesToPrint:
        #print("~~~~~~~~~~~~~~~~~~~~~ \n Day to find = ",day)
        count = 0
        if day == datesToPrint[0]:
            for row in data:
                if row['Date'] == day:
                        #print(f"Date: {row['Date']}")
                        print(f"{row['Open']}\n{row['High']}\n{row['Low']}\n{row['Adj Close']}\n{row['Volume']}")
                        break
        else:
            for row in data:
                if row['Date'] == day:
                        #print(f"Date: {row['Date']}")
                        print(f"{row['Adj Close']}")
                        break
                else:
                        excelDate = row['Date']
                        #print("current row date= ",excelDate)
                        try:
                            excelDate = datetime.datetime.strptime(excelDate, "%Y-%m-%d")
                        except:
                            excelDate = datetime.datetime.strptime(excelDate, '%d/%m/%Y')

                        dateLookingFor = day
                        #print("date looking for= ", dateLookingFor)
                        try:
                            dateLookingFor = datetime.datetime.strptime(dateLookingFor, "%Y-%m-%d")
                        except:
                            dateLookingFor = datetime.datetime.strptime(dateLookingFor, '%d/%m/%Y')

                        if excelDate > dateLookingFor:
                            count2 = 0
                            for row in data:
                                if count-1 == count2:
                                    #print(f"Current date = {row['Date']}") #finding the date before the date that doesn't exist
                                    #print(f"Date: {row['Date']}")
                                    print(f"{row['Adj Close']}")
                                    break
                                else:
                                    count2+=1
                            break
                        else:
                            count += 1
def openFileToWrite(mainFileName):
    with open(mainFileName, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

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

def moveDataCsv(data):
    target_folder = os.getcwd() + "\Data Csvs"
    for i in os.listdir():
        if i.endswith('.csv'):
            source_path = os.path.join(os.getcwd(), i)
            target_path = os.path.join(target_folder, i)
            # move the file to the target folder
            #print(source_path, "\n", target_path)
            shutil.move(source_path, target_path)
            #fix my error
def main():
        data = openFileToRead()
        # Get the start date
        startDateStr = data[0]['Date']
        #print(startDateStr)
        try:
            datetimeObj = datetime.datetime.strptime(startDateStr, "%Y-%m-%d")
        except:
            datetimeObj = datetime.datetime.strptime(startDateStr, "%d/%m/%Y")
        #print(datetimeObj)
        dayList = [0,7,30,90,180,365,730,1825,3650]
        dayListAlt = adjustDayList(dayList)
        datesToPrint = endDateCalculate(datetimeObj,dayList)

        #openFileToWrite(mainFileName)
        #print(datesToPrint)
        printDataOnDays(datesToPrint,data)
        moveDataCsv(data)


while True:
    try:
        main()
    except:
        pass