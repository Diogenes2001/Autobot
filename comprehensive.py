from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

import tensorflow as tf
from tensorflow import keras
""" from tensorflow.keras import layers """
""" 
print(tf.__version__) """

dataset_path = "file://localhost/C:/Users/ethan/Documents/Python/training_data_large.csv"
dataset_path

""" "DRIVING_EXPERIENCE", "YEARS_SINCE_AT_FAULT_CLAIM","YEARS_SINCE_NOT_AT_FAULT_CLAIM","YEARS_SINCE_MINOR_CONVICTION","YEARS_SINCE_MAJOR_CONVICTION","YEARS_SINCE_SERIOUS_CONVICTION" """

column_names = ['LATITUDE', 'LONGITUDE', 'AGE', 'ANNUAL_KILOMETERS', 'DAILY_KILOMETERS', 'VEHICLE_YEAR',
                "NUMBER_OF_DRIVERS", "NUMBER_OF_VEHICLES", 'DEDUCTIBLE_COMPREHENSIVE',
                'AT_FAULT_CLAIMS', 'NOT_AT_FAULT_CLAIMS', "MINOR_CONVICTIONS", "MAJOR_CONVICTIONS", "SERIOUS_CONVICTIONS",
                "GENDER","DRIVER_MARTIAL_STATUS", 'VEHICLE_AGE', 
                'INCURRED_LOSS_COMPREHENSIVE']
raw_dataset = pd.read_csv(dataset_path, 
                      usecols=column_names, 
                      na_values = "?", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

dataset = dataset.dropna()
dataset = dataset[dataset['INCURRED_LOSS_COMPREHENSIVE'] > 0]

""" print(dataset.head(10)) """

gender = dataset.pop('GENDER')
dataset['MALE'] = (gender == 'M')*1.0
dataset['FEMALE'] = (gender == 'F')*1.0

maritalstatus = dataset.pop('DRIVER_MARTIAL_STATUS')
dataset['SINGLE'] = (maritalstatus == 'S')*1.0
dataset['MARRIED'] = (maritalstatus == 'M')*1.0



train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

""" sns.pairplot(train_dataset[["INCURRED_LOSS_COMPREHENSIVE", "AGE", "VEHICLE_YEAR"]], diag_kind="kde")
plt.show() """

train_stats = train_dataset.describe()
train_stats.pop("INCURRED_LOSS_COMPREHENSIVE")
train_stats = train_stats.transpose()
train_stats


train_labels = train_dataset.pop('INCURRED_LOSS_COMPREHENSIVE')
test_labels = test_dataset.pop('INCURRED_LOSS_COMPREHENSIVE')

""" def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset) """

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()

example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
example_result

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 100

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

""" test_predictions = model.predict(test_dataset).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [INCURRED_LOSS_COMPREHENSIVE]')
plt.ylabel('Predictions [INCURRED_LOSS_COMPREHENSIVE]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

plt.show()
 """
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
print("Testing set Mean Abs Error: ${:5.2f}".format(mae))
print("Testing set Mean Root Squared Error: $" + str(round(math.sqrt(mse), 2)))

model.save('comprehensive.h5')
