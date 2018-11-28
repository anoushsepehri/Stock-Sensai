import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
 
 
#plt.switch_backend('QT5Agg')  
 
 
 
dates = []
prices = []
 
def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) # skipping column names
        for row in csvFileReader:
            #dates.append(int(row[0].split('-')[0]))
            #dates.append(float(row[0]))
            prices.append(float(row[1]))
    return
 
def predict_price(dates, prices, x):

    array=[]
    for i in range(x):
        array.append(i)

    dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1
    array = np.reshape(array,(len(array), 1)) # converting to matrix of n X 1
 
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
    svr_rbf.fit(dates, prices) # fitting the data points in the models
    predicted_price=svr_rbf.predict(array)
   # print("Stock Predicted Price: ",predicted_price[-1])
 
    plt.scatter(dates, prices, color= 'black', label= 'Previous Prices') # plotting the initial datapoints 
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
    plt.plot(array, predicted_price, color= 'red', label= 'Predicted Price') # plotting the line made by the RBF kernel
    plt.scatter(array, predicted_price, color= 'red', label= 'Predicted Price') # plotting the line made by the RBF kernel
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression Model')
    plt.legend()
    plt.show()
 
    #return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]
    return svr_rbf.predict(array)
 
 
get_data('SPLK.csv') # calling get_data method by passing the csv file to it
x=len(prices)

for i in range(x):
    dates.append(i)

#print ("Dates- ", dates)
#print ("Prices- ", prices)
 
predicted_price = predict_price(dates, prices, x+1)  
print (predicted_price[-1])
