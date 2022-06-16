import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 将csv数据读入pandas dataframe
file_path = 'fer2013.csv'
df = pd.read_csv(file_path)


# 将数据集划分为训练和测试模块
X_train, Y_train, X_test, Y_test = [], [], [], []

for index, row in df.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val, 'float32'))
            Y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val, 'float32'))
            Y_test.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

num_features = 64
num_labels = 7
batch_size = 64
epochs = 200
width, height = 48, 48

X_train = np.array(X_train, 'float32')
Y_train = np.array(Y_train, 'float32')
X_test = np.array(X_test, 'float32')
Y_test = np.array(Y_test, 'float32')

# 将0-6的标签数据转化为0-1的类别数据
Y_train = np_utils.to_categorical(Y_train, num_classes=num_labels)
Y_test = np_utils.to_categorical(Y_test, num_classes=num_labels)

# normalizing data between 0 and 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# 构建CNN
model = Sequential()

# 第一层卷积层
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

# 第二层卷积层
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

# 第三层卷积层
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

# 全连接层
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

# 分为要输出的几种种类
model.add(Dense(num_labels, activation='softmax'))
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

# 训练模型
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(X_test, Y_test), shuffle=True)

# 保存训练数据
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
