import pandas as pd
import numpy as np
from sklearn.svm import SVR

def get_data(doc):
	data=pd.read_csv(doc)
	open_price=data.Open
	dates=data.Date
	dates=dates.tolist()
	open_price=open_price.tolist()
	train_arr=[dates,open_price]
	return train_arr

def fit_regression(train_arr):
	training=[]
	dates=train_arr[0]
	open_prices=train_arr[1]

	for i in range(len(dates)):
		training.append(i)

	training = np.reshape(training,(len(training), 1))
	open_prices = np.reshape(open_prices,(len(open_prices), 1))

	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
	svr_rbf.fit(training, open_prices.ravel())
	next_day=len(dates)
	prediction=svr_rbf.predict([[next_day]])

	return prediction


doc=input("Which csv file would you like to analyze: ")

train_arr=get_data(doc)
prediction=fit_regression(train_arr)

end=input("Tomorrow's Opening Stock Price is " + str(prediction[0]))

