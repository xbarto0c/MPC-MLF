import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras import regularizers
from keras.datasets import cifar10
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, StandardScaler
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Load the input data
x_train = pd.read_csv(r'x_train.csv');
y_train = pd.read_csv(r'y_train.csv');
x_test = pd.read_csv(r'x_test.csv');

# print(x_train);
# print(y_train);
# print(x_test);

# Dump the unwanted features
x_train = x_train.drop(columns = ['m_power', 'Tosc', 'Tmix']);
x_test = x_test.drop(columns = ['m_power', 'Tosc', 'Tmix']);
# Delete colums that couldn't be deleted in the previous step
del x_train[x_train.columns[0]];
del x_test[x_test.columns[0]];

# Perform standardization on the input data 
scaler = StandardScaler();
x_train_scaled = scaler.fit_transform(x_train);
x_test_scaled = scaler.fit_transform(x_test);

# Min-max scaling the training data - not used
""" x_train_scaled = x_train.copy();
for column in x_train_scaled.columns:
    x_train_scaled[column] = x_train_scaled[column]  / x_train_scaled[column].abs().max();

print(x_train_scaled);



# Min-max scaling the testing data
x_test_scaled = x_test.copy();
for column in x_test_scaled.columns:
    x_test_scaled[column] = x_test_scaled[column]  / x_test_scaled[column].abs().max(); """

# Convert scaled input data into pandas dataframe for the model to be able to read them
x_test_scaled = pd.DataFrame(x_test_scaled);
x_train_scaled = pd.DataFrame(x_train_scaled);


# print(x_test.shape);

# Delete the id column from ground truth values
y_train = y_train.drop(columns = "id");
# Convert the ground truth values into a numpy array
y_train = y_train.to_numpy();

# print(y_train);

y_train = y_train - 1; # the categories need to start from 0
# Conver the ground truth data into categorical data
y_train_encoded = to_categorical(y_train, num_classes=8);

#print(y_train_encoded);

# Initialise early stopping feature
early_stopping = EarlyStopping(monitor='val_loss', patience = 15);
# Define the model architecture
model = Sequential();
model.add(Flatten(input_shape=(len(x_train_scaled.axes[1]),))); # Flatten the input data
model.add(Dropout(0.1)); # Adding a dropout layer

model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3),
    bias_regularizer=regularizers.L2(1e-3), activity_regularizer=regularizers.L2(1e-4))); # Defining the hidden layer

model.add(Dense(8, activation='softmax')); # Defining the output layer

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy']);
model.summary();
# Train the model
history = model.fit(x_train_scaled, y_train_encoded, epochs=800, batch_size=5, validation_split = 0.4, callbacks = [early_stopping]);

# Calculate the performance of the model
score = model.evaluate(x_train_scaled, y_train_encoded, verbose=0)
print('Test loss:', score[0])
print(f'Test accuracy: {score[1]*100} %')

# Use the model to predict the types of devices
predicted_data = model.predict(x_test_scaled);
data_classes = predicted_data.argmax(axis=-1) + 1; # the categories need to start from 0 -> revert this step

# Convert the predicted data into pandas dataframe and save them in the required format
df = pd.DataFrame(data_classes, columns = ['"target"']);
df.index.name = '"id"';
df.to_csv("output.csv");

# Plot the histogram and performance of the model
plt.subplot(3,1,1);
plt.hist(df['"target"'], color='blue', edgecolor='black', bins=int(45/1));
 
plt.xlabel('Device number');
plt.ylabel('No. of devices');
plt.title('Predicted types of devices');


plt.subplot(3,1,2);
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Number of epochs [-]');
plt.ylabel('Loss [-]');
plt.title('Prediction losses');
plt.legend(['Loss', 'Validation loss']);
#plt.show();


plt.subplot(3,1,3);
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Number of epochs [-]');
plt.ylabel('Accuracy [-] [-]');
plt.title('Accuracy of the predictions');
plt.legend(['Accuracy', 'Validation accuracy']);
plt.show();




