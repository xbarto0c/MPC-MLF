import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from PIL import Image

x_train = pd.read_csv(r'Dataset\Train\CSV\img_0.csv'); # read the first csv file
x_test = pd.read_csv(r'Dataset\Test\CSV\img_0.csv');
y_train = pd.read_csv(r'Dataset\y_train.csv')
del x_train[x_train.columns[0]]; # delete the first ID column
del x_test[x_test.columns[0]];
x_train = x_train.to_numpy(); # convert to numpy array
x_test = x_test.to_numpy();


for i in range(1, 4000):
    x_train_temp = pd.read_csv(r'Dataset\Train\CSV\img_'+str(i)+'.csv'); # read the remaining files and join them together
    del x_train_temp[x_train_temp.columns[0]];
    x_train = np.append(x_train, x_train_temp.to_numpy());
    print(i);

for i in range(4000, 8279):
    x_train_temp = pd.read_csv(r'Dataset\Train\CSV\img_'+str(i)+'.csv'); # read the remaining files and join them together
    del x_train_temp[x_train_temp.columns[0]];
    x_train_help = np.append(x_train_help, x_train_temp.to_numpy());
    print(i);
x_train = np.append(x_train, x_train_help);

for i in range(1, 3549):
    x_test_temp = pd.read_csv(r'Dataset\Test\CSV\img_'+str(i)+'.csv');
    del x_test_temp[x_test_temp.columns[0]];
    x_test = np.append(x_test, x_test_temp.to_numpy());
    print(i);


# scaler = MinMaxScaler(); 3549
# x_train = scaler.fit_transform(x_train);
# x_test = scaler.fit_transform(x_test);

# np.savetxt("x_train.csv", x_train, delimiter=",");
# np.savetxt("x_test.csv", x_test, delimiter=",");
x_train = x_train.reshape(8279, 44, 51, 1); # reshape the train and test data
x_test = x_test.reshape(3549, 44, 51, 1);

y_train = y_train.drop(columns="id"); # preprocess the ground truth data
y_train = y_train.to_numpy();
y_train = y_train - 1  # the categories need to start from 0
y_train_encoded = to_categorical(y_train, num_classes=3);
# print(y_train_encoded);
# y_test = to_categorical(y_test);

# convert from integers to floats
x_train = x_train.astype('float16');
x_test = x_test.astype('float16');
# normalize to range 0-1
print(np.max(x_train));
print(np.max(x_test));
# df = pd.DataFrame(x_train.reshape(51,156156));
# df.to_csv("train.csv");
x_train = x_train / 255; # normalise the test and train data
x_test = x_test / 255;
print(np.max(x_train));
print(np.max(x_test));
# scaler = StandardScaler();
# x_train_scaled = scaler.fit_transform(x_train);
# x_test_scaled = scaler.fit_transform(x_test);

# print(x_test);
# print(x_train);

model = Sequential(); # build the model
early_stopping = EarlyStopping(monitor='val_loss', patience=15);
model.add(Conv2D(32, strides=(4, 4), kernel_size=(10, 10), activation='relu',
              kernel_initializer='he_uniform', input_shape=(682, 539, 1)));
model.add(MaxPooling2D((2, 2)));
model.add(Flatten());
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'));
model.add(Dense(3, activation='softmax'));
# compile model
# opt = SGD(learning_rate=0.01, momentum=0.9);
opt = Adam(learning_rate=0.01);
model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy']);
model.summary();
history = model.fit(x_train, y_train_encoded, epochs=80,
                        batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping]);
model.save('model');

# plot the results
pt.subplot(2, 1, 1);
pt.title('Cross Entropy Loss');
pt.plot(history.history['loss'], color='blue', label='train');
pt.plot(history.history['val_loss'], color='orange', label='test');
# plot accuracy
pt.subplot(2, 1, 2);
pt.title('Classification Accuracy');
pt.plot(history.history['accuracy'], color='blue', label='train');
pt.plot(history.history['val_accuracy'], color='orange', label='test');
pt.show();

predicted_data = model.predict(x_test);
data_classes = predicted_data.argmax(axis=-1) + 1;

df = pd.DataFrame(data_classes, columns=['"target"']);
df.index.name = '"id"';
df.to_csv("output.csv");

pt.hist(df['"target"'], color='blue', edgecolor='black', bins=int(45/1));

pt.xlabel('Device number');
pt.ylabel('No. of devices');
pt.title('Predicted types of devices');
pt.show();
