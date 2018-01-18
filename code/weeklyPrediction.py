# RNN prediction for each week testing.
# Run on hourly dataset which contain 168 data per week

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

lookBack = 168
epoch = 500
batchSize = 1

# load the dataset
dataframe = pandas.read_csv('ActData-9weeks.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

totalData = int(len(dataframe))
weeks = int(totalData/lookBack)
numberOfRun = weeks - 1

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=lookBack):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# split into train and test sets

for x in range(2,weeks):
	train_size = lookBack*x
	test_size = lookBack*(x+1)
	train, test = dataset[0:train_size,:], dataset[(train_size-lookBack):(train_size+lookBack),:]

	trainX, trainY = create_dataset(train, lookBack)
	testX, testY = create_dataset(test, lookBack)

	# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(168, input_shape=(1, lookBack), activation='tanh'))
	model.add(Dense(128, activation='tanh'))
	model.add(Dense(32, activation='tanh'))	
	model.add(Dense(1, activation='tanh'))	
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)

	# numpy.savetxt('trainData'+str(x)+'.csv', train, delimiter=",")
	# numpy.savetxt('trainXData'+str(x)+'.csv', trainX, delimiter=",")
	# numpy.savetxt('trainYData'+str(x)+'.csv', trainY, delimiter=",")
	# numpy.savetxt('testData'+str(x)+'.csv', test, delimiter=",")
	# numpy.savetxt('testXData'+str(x)+'.csv', testX, delimiter=",")
	# numpy.savetxt('testYData'+str(x)+'.csv', testY, delimiter=",")

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	# calculate root mean squared error
	trainScore = mean_absolute_error(trainY[0], trainPredict[:,0])
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = mean_absolute_error(testY[0], testPredict[:,0])
	print('Test Score: %.2f RMSE' % (testScore))

	# shift train for plotting
	trainPredictPlot = numpy.empty_like(dataset)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[lookBack:len(trainPredict)+lookBack, :] = trainPredict

	# shift test for plotting
	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+lookBack:len(trainPredict)+(2*lookBack), :] = testPredict

	numpy.savetxt('trainResultAct'+str(x)+'.csv', trainPredictPlot, delimiter=",")
	numpy.savetxt('testResultAct'+str(x)+'.csv', testPredictPlot, delimiter=",")

	# plotting
	# plt.plot(scaler.inverse_transform(dataset))
	# plt.plot(trainPredictPlot)
	# plt.plot(testPredictPlot)
	# plt.show()