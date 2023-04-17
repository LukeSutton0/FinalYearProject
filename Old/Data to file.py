import csv


# Read the CSV file into a list of dictionaries
with open('CTA.L .csv', 'r') as file:
    reader = csv.DictReader(file)
    data = [row for row in reader]

# Define a function to print the data on certain days
def printDataOnDays(days):
    for i, row in enumerate(data):
        if i == 0:
            print("Start")
            print(f"Date: {row['Date']}, Open: {row['Open']}, High: {row['High']}, Low: {row['Low']}, Close: {row['Close']}, Adj Close: {row['Adj Close']}, Volume: {row['Volume']}")
        elif i+1 in days:
            print(f"Date: {row['Date']}, Open: {row['Open']}Adj Close: {row['Adj Close']}")

# Example usage

def getFirstDay():
    first_date = data[0]['Date']
    return first_date

printDataOnDays([30, 90, 180, 365])

firstDateVal = getFirstDay()
print(firstDateVal)
#printDataOnDays

