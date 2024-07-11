import numpy as np
np.random.seed(10)
import random 
random.seed(10)
import tensorflow as tf
tf.random.set_seed(10)


from glob import glob

#glob() 回傳符合條件的檔案名稱列表
imagePaths = []
for i in range(0,219):
    imagePaths.extend(glob('category2\\'+str(i) +'\\*.jpg')[:-2])
    
testImagePaths = []
for i in range(0,219):
    testImagePaths.extend(glob('category2\\'+str(i) +'\\*.jpg')[-2:])
#testImagePaths.extend(glob('orchid_public_set\\*.jpg')[:])
#testImagePaths.extend(glob('orchid_private_set\\*.jpg')[:])

#使用 Python 字串的 split() 方法來分割路徑，-2
labels = [image.split('\\')[-2] for image in imagePaths]
#%%

#超參數
resize = 224
batch_size = 4
epochs = 30


import cv2 #OpenCV用以讀入圖片
import numpy as np #Numpy用以處理陣列資料
from tqdm import tqdm #顯示進度條方停確認讀取進度

#初始化圖片陣列
images = np.zeros((len(imagePaths), resize, resize, 3), np.uint8)
test_images = np.zeros((len(testImagePaths), resize, resize, 3), np.uint8)

#使用廻圈逐一輸圖片存入陣列
idx = 0
for path in tqdm(imagePaths):
  img = cv2.imread(path) #讀取圖片
  img = cv2.resize(img, (resize, resize)) #resize成resize*resize
  images[idx] = img #將圖片存入陣列
  idx += 1
  
#使用廻圈逐一輸測試圖片存入陣列
idx = 0
for path in tqdm(testImagePaths):
  img = cv2.imread(path) #讀取圖片
  img = cv2.resize(img, (resize, resize)) #resize成resize*resize
  test_images[idx] = img #將圖片存入陣列
  idx += 1
#%%  
  
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50
from keras.applications.resnet import ResNet101
from keras.applications.resnet import ResNet152
from keras.applications.efficientnet_v2 import EfficientNetV2L
from keras.applications.efficientnet import EfficientNetB0
#from tensorflow.keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
#取得所有類別
# classes = []
# for i in labels:
#     if(i not in classes):
#         classes.append(i)
classes = np.unique(labels)
print(classes)
#產生類別對應字典
class_map = {
    cls: i for i, cls in enumerate(classes)
}
print('classMap = ', class_map)
#轉換成以數字標記
enc = LabelEncoder()
labels = enc.fit_transform(labels)
#One-hot encoding
labels = keras.utils.to_categorical(labels)
#%%

# def preproc(images):
#     #image_byte = tf.io.read_file(fpath)
#     #image = tf.io.decode_image(image_byte)
#     #image_resize = tf.image.resize_with_pad(image, resize, resize) #缩放到224*224
#     image_norm = tf.cast(images, tf.float32) / 255. #归一化
#     images = image_norm
#     images = tf.image.random_brightness(images, 0.10)
#     images = tf.image.random_flip_left_right(images)
#     images = tf.image.random_contrast(images,0.3,0.7)
#     images = tf.image.random_hue(images,0.1)
    
#     return images

x_train, x_val, y_train, y_val = train_test_split(
    images, labels,
    test_size=0.2,
    random_state=0,
    stratify=labels 
)
#%%

def load_model():
  base_network = ResNet50(include_top=False, weights='imagenet', input_shape=(resize, resize, 3))
  #凍結預設的參數
  for layer in base_network.layers:
    layer.trainable = True #設定為True,即每層參數可重新訓練; False則為鎖定每層參數，不重新訓練

  #在預訓練模型後，接上自訂之全連階層
  x = base_network.output
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)
#  x = Dense(1024, activation='relu')(x)
#  x = Dropout(0.5)(x)
  x = Dense(512, activation='relu')(x)
  x = Dropout(0.5)(x)
  predictions = Dense(219, activation='softmax')(x) #最終層輸出為219個類別
  model = Model(inputs=base_network.input, outputs=predictions)

  return model
#%%

#載人模型
model = load_model()
#model.summary()
#%%

from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

escb = EarlyStopping(monitor="val_loss",
            patience=3,
            verbose=1,
            restore_best_weights=True)

lrcb = ReduceLROnPlateau(monitor="val_loss",
              factor=0.95,
              patience=2,
              verbose=1,
              min_lr=0.001)
#%%
from keras.preprocessing.image import  ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.1,
        fill_mode='nearest',
        width_shift_range=0.1,
        horizontal_flip=True,
        height_shift_range=0.1)
#定義 loss function
#sgd = SGD(learning_rate=0.005, decay=1e-5)
optimizer = tf.optimizers.SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#開始訓練
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                              validation_data=(x_val, y_val), verbose=1,
                              steps_per_epoch=x_train.shape[0]//batch_size,
                              callbacks=[escb, lrcb],
                              shuffle=True)

#輸出結果
print('Train Acc:', model.evaluate(x_train, y_train)[1])
print('Test Acc:', model.evaluate(x_val, y_val)[1])
#%%

#model.save(r'1080617.h5') #把模型名稱取為你的學號
#%%

import matplotlib.pyplot as plt
f = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#%%

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sn
# from sklearn.metrics import confusion_matrix

# #取得測試集的實際類別和預測類別
# y_true = np.argmax(y_test, axis=1)
# y_pred = model.predict(x_test).argmax(axis=1)

# #產生混淆矩陣
# df_cm = pd.DataFrame(confusion_matrix(y_true, y_pred),
#             index=classes,
#             columns=classes)
# cmFig = plt.figure(figsize=(10, 7))
# sn.set(font_scale=2)
# sn.heatmap(df_cm, annot=True, annot_kws={"size":32}, fmt="d")
# plt.title('Confusion Matrix', fontsize=24)
# plt.xlabel('True Label', fontsize=18)
# plt.ylabel('Prediction Label', fontsize=18)
#%%
#儲存模型
#model = tf.keras.models.load_model('1080617.h5')

import numpy as np
import pandas as pd
#預測test資料並寫入csv
prediction = model.predict(test_images).argmax(axis=1)
#y_pred = model.predict(test_images).argmax(axis=1)
sample = pd.read_csv('1080617.csv')
print(prediction)

new_prediction = []
new_class_map = {v : k for k, v in class_map.items()}
#40285
#41425
for i in range(438):
  new_prediction.append(new_class_map[prediction[i]])
#sample['file'] = testImagePaths
sample['filename'] = [label.split('\\')[-1]     for label in testImagePaths]
sample['category'] = new_prediction
sample.to_csv('1080617.csv', header=True, index=False)

#計算test_data準度
label = [i.split('\\')[-2] for i in testImagePaths]
equ_num = 0
for i in range(438):
    if (label[i] == new_prediction[i]):
        equ_num = equ_num + 1
print(equ_num/438)