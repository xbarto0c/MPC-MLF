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

x_train = Image.open('Dataset\Train\BW\img_0.png').convert('L'); # read first BW image from train dataset
x_test = Image.open('Dataset\Test\BW\img_0.png').convert('L'); # read first BW image from test dataset
y_train = pd.read_csv(r'Dataset\y_train.csv'); # read ground truth values

print(x_test.size);
# Image loading is done with support variables to speed up the process
for i in range(1, 500):
    x_train = np.append(x_train, Image.open('Dataset\Train\BW\img_'+str(i)+'.png').convert('L')); # read images number 1 - 499 and append them to numpy array
    x_test = np.append(x_test, Image.open('Dataset\Test\BW\img_'+str(i)+'.png').convert('L'));# read images number 1 - 499 and append them to numpy array

    print(i);

x_train_1 = Image.open('Dataset\Train\BW\img_500.png').convert('L');
x_test_1 = Image.open('Dataset\Test\BW\img_500.png').convert('L');
for i in range(500, 1000):
    x_train_1 = np.append(x_train_1, Image.open('Dataset\Train\BW\img_'+str(i)+'.png').convert('L'));
    x_test_1 = np.append(x_test_1, Image.open('Dataset\Test\BW\img_'+str(i)+'.png').convert('L'));

    print(i);

x_train_2 = Image.open('Dataset\Train\BW\img_1000.png').convert('L');
x_test_2 = Image.open('Dataset\Test\BW\img_1000.png').convert('L');
for i in range(1001, 1500):
    x_train_2 = np.append(x_train_2, Image.open('Dataset\Train\BW\img_'+str(i)+'.png').convert('L'));
    x_test_2 = np.append(x_test_2, Image.open('Dataset\Test\BW\img_'+str(i)+'.png').convert('L'));

    print(i);
x_train_3 = Image.open('Dataset\Train\BW\img_1500.png').convert('L');
x_test_3 = Image.open('Dataset\Test\BW\img_1500.png').convert('L');
for i in range(1500, 2000):
    x_train_3 = np.append(x_train_3, Image.open('Dataset\Train\BW\img_'+str(i)+'.png').convert('L'));
    x_test_3 = np.append(x_test_3, Image.open('Dataset\Test\BW\img_'+str(i)+'.png').convert('L'));

    print(i);
x_train_4 = Image.open('Dataset\Train\BW\img_2000.png').convert('L');
x_test_4 = Image.open('Dataset\Test\BW\img_2000.png').convert('L');
for i in range(2001, 2500):
    x_train_4 = np.append(x_train_4, Image.open('Dataset\Train\BW\img_'+str(i)+'.png').convert('L'));
    x_test_4 = np.append(x_test_4, Image.open('Dataset\Test\BW\img_'+str(i)+'.png').convert('L'));

    print(i);
x_train_5 = Image.open('Dataset\Train\BW\img_2500.png').convert('L');
x_test_5 = Image.open('Dataset\Test\BW\img_2500.png').convert('L');
for i in range(2501, 3000):
    x_train_5 = np.append(x_train_5, Image.open('Dataset\Train\BW\img_'+str(i)+'.png').convert('L'));
    x_test_5 = np.append(x_test_5, Image.open('Dataset\Test\BW\img_'+str(i)+'.png').convert('L'));

    print(i);
x_train_6 = Image.open('Dataset\Train\BW\img_3000.png').convert('L');
x_test_6 = Image.open('Dataset\Test\BW\img_3000.png').convert('L');
for i in range(3001, 3500):
    x_train_6 = np.append(x_train_6, Image.open('Dataset\Train\BW\img_'+str(i)+'.png').convert('L'));
    x_test_6 = np.append(x_test_6, Image.open('Dataset\Test\BW\img_'+str(i)+'.png').convert('L'));

    print(i);
x_train_7 = Image.open('Dataset\Train\BW\img_3500.png').convert('L');
x_test_7 = Image.open('Dataset\Test\BW\img_3500.png').convert('L');
for i in range(3501, 3549):
    x_train_7 = np.append(x_train_7, Image.open('Dataset\Train\BW\img_'+str(i)+'.png').convert('L'));
    x_test_7 = np.append(x_test_7, Image.open('Dataset\Test\BW\img_'+str(i)+'.png').convert('L'));

    print(i);

# scaler = MinMaxScaler(); 3549
# x_train = scaler.fit_transform(x_train);
# x_test = scaler.fit_transform(x_test);
x_train = np.append(x_train, x_train_1); # merge all the variables into one huge
x_train = np.append(x_train, x_train_2);
x_train = np.append(x_train, x_train_3);
x_train = np.append(x_train, x_train_4);
x_train = np.append(x_train, x_train_5);
x_train = np.append(x_train, x_train_6);
x_train = np.append(x_train, x_train_7);
x_test = np.append(x_test, x_test_1);
x_test = np.append(x_test, x_test_2);
x_test = np.append(x_test, x_test_3);
x_test = np.append(x_test, x_test_4);
x_test = np.append(x_test, x_test_5);
x_test = np.append(x_test, x_test_6);
x_test = np.append(x_test, x_test_7);
#np.savetxt("x_train.csv", x_train, delimiter=",");
#np.savetxt("x_test.csv", x_test, delimiter=",");
x_train = x_train.reshape(3549, 682, 539, 1); # reshape the array into something the model can work with
x_test = x_test.reshape(3549, 682, 539, 1);


y_train = y_train.drop(columns = "id"); # preprocess the ground truth values
y_train = y_train.to_numpy();
y_train = y_train - 1; # the categories need to start from 0
y_train_encoded = to_categorical(y_train, num_classes=3);
# print(y_train_encoded);
# y_test = to_categorical(y_test);

# convert from integers to floats
x_train = x_train.astype('float16');
x_test = x_test.astype('float16');
# normalize to range 0-1
print(np.max(x_train));
print(np.max(x_test));
#df = pd.DataFrame(x_train.reshape(51,156156));
#df.to_csv("train.csv");
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
early_stopping = EarlyStopping(monitor='val_loss', patience = 15);
model.add(Conv2D(32, strides = (4, 4), kernel_size = (10,10), activation='relu', kernel_initializer='he_uniform', input_shape=(682, 539, 1)));
model.add(MaxPooling2D((2, 2)));
model.add(Flatten());
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'));
model.add(Dense(3, activation='softmax'));
# compile model
# opt = SGD(learning_rate=0.01, momentum=0.9);
opt = Adam(learning_rate=0.01);
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']);
model.summary();
history = model.fit(x_train, y_train_encoded, epochs=80, batch_size=32, validation_split=0.2, verbose=1, callbacks = [early_stopping]);

# plot the results
pt.subplot(2, 1, 1)
pt.title('Cross Entropy Loss')
pt.plot(history.history['loss'], color='blue', label='train')
pt.plot(history.history['val_loss'], color='orange', label='test')
# plot accuracy
pt.subplot(2, 1, 2)
pt.title('Classification Accuracy')
pt.plot(history.history['accuracy'], color='blue', label='train')
pt.plot(history.history['val_accuracy'], color='orange', label='test')
pt.show()

predicted_data = model.predict(x_test);
data_classes = predicted_data.argmax(axis=-1) + 1;

df = pd.DataFrame(data_classes, columns = ['"target"']);
df.index.name = '"id"';
df.to_csv("output.csv");

pt.hist(df['"target"'], color='blue', edgecolor='black', bins=int(45/1));
 
pt.xlabel('Device number');
pt.ylabel('No. of devices');
pt.title('Predicted types of devices');
pt.show()
