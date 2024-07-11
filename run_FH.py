import os, shutil, random, glob
import copy
#添加tensorflow的库
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#添加matplotlib的库进行界面可视化
import matplotlib.pyplot as plt

from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import ResNet152
from keras.applications.inception_v3 import InceptionV3
from keras.applications.efficientnet import EfficientNetB0
 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
#
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#变量
resize = 224  #输入图片尺寸参数
#total_num = 2190  #训练集总计和
#train_num = 1971  #训练集实际训练数字，20的验证集
epochs = 50       #迭代次数
batch_size = 4   #每次训练多少张

# -----------------------------------------------------------------------------
dir_data = 'C:\\flower_category'           #训练集路径

dir_category = os.path.join(dir_data, 'category2')
category_list = os.listdir(dir_category)
category_list.sort(key=lambda x:int(x.split('.')[0]))

# #健壮性的断言
# #assert os.path.exists(dir_mask), 'Could not find ' + dir_mask
# #assert os.path.exists(dir_nomask), 'Could not find ' + dir_nomask
# #定义了文件指针对整个文件夹遍历一遍，将图像读出来

i = 0
for c in category_list:
    category_list[i] = [os.path.abspath(fp) for fp in glob.glob(os.path.join(os.path.join(dir_category, c), '*.jpg'))]
    i = i+1

# fpath_nomask = [os.path.abspath(fp) for fp in glob.glob(os.path.join(dir_nomask, '*.jpg'))]
# #文件数
num_data = len(category_list[0])
# num_nomask = len(fpath_nomask)
# #设置标签
label_data = copy.copy(category_list)
for i in range(219):
    label_data[i] = [i] * num_data 


#label_mask = [0] * num_data 
#label_nomask = [1] * num_data 
# 
# print('#mask:   ', num_mask)
# print('#nomask: ', num_nomask)
# #划分多少为验证集
RATIO_TEST = 0.1
# 
num_data_test = int(num_data * RATIO_TEST)
# num_nomask_test = int(num_nomask * RATIO_TEST)
# 
# # train
fpath_train = []
for i in range(219):
    fpath_train = fpath_train + category_list[i][num_data_test:]

label_train = []
for i in range(219):
    label_train = label_train + label_data[i][num_data_test:]

# validation

fpath_vali = []
for i in range(219):
    fpath_vali = fpath_vali + category_list[i][:num_data_test]
#fpath_vali = fpath_mask[:num_data_test] + fpath_nomask[:num_nomask_test]

label_vali = []
for i in range(219):
    label_vali = label_vali + label_data[i][:num_data_test]
#label_vali = label_mask[:num_data_test] + label_nomask[:num_nomask_test]
# 
num_train = len(fpath_train)
num_vali = len(fpath_vali)
# 
print(num_train)
print(num_vali)
# =============================================================================
#预处理
def preproc(fpath, label):
    image_byte = tf.io.read_file(fpath)
    image = tf.io.decode_image(image_byte)
    image_resize = tf.image.resize_with_pad(image, resize, resize) #缩放到224*224
    image_norm = tf.cast(image_resize, tf.float32) / 255. #归一化
    image = image_norm
    image = tf.image.random_brightness(image, 0.10)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image,0.3,0.7)
    image = tf.image.random_hue(image,0.1)
    
    label_onehot = tf.one_hot(label, 219)
    
    return image, label_onehot

dataset_train = tf.data.Dataset.from_tensor_slices((fpath_train, label_train)) #将数据进行预处理
dataset_train = dataset_train.shuffle(num_train).repeat()  #打乱顺序
dataset_train = dataset_train.map(preproc, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset_train = dataset_train.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE) #一批次处理多少份

dataset_vali = tf.data.Dataset.from_tensor_slices((fpath_vali, label_vali))
dataset_vali = dataset_vali.shuffle(num_vali).repeat()
dataset_vali = dataset_vali.map(preproc, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset_vali = dataset_vali.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
# -----------------------------------------------------------------------------

model = keras.Sequential()
model.add(EfficientNetB0(include_top=False, pooling='avg', weights='imagenet'))
model.add(layers.Flatten())
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(219, activation='softmax'))
model.summary

filepath='C:/flower_category/model'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True,mode='max',period=2) 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.95,
                              patience=2,verbose=1, min_lr=0.001)
escb = EarlyStopping(monitor="val_loss",patience=3,verbose=1,
                     restore_best_weights=True)

callbacks_list = [checkpoint,reduce_lr,escb]
  
model.compile(loss='categorical_crossentropy', 
              optimizer='sgd',
              metrics=['acc'])


history = model.fit(dataset_train,
          steps_per_epoch = num_train//batch_size,
          epochs = epochs,         #迭代次数
          validation_data = dataset_vali,
          validation_steps = num_vali//batch_size,
          verbose = 1,
          callbacks=callbacks_list)
#评分标准
# scores = model.evaluate(train_data, train_label, verbose=1)
scores = model.evaluate(dataset_train, steps=num_train//batch_size, verbose=1)
print(scores)

# scores = model.evaluate(test_data, test_label, verbose=1)
scores = model.evaluate(dataset_vali, steps=num_vali//batch_size, verbose=1)
print(scores)
#保存模型
#model.save('C:/flower_category/model/ResNet50.h5')

# Record loss and acc
history_dict = history.history
train_loss = history_dict['loss']
train_accuracy = history_dict['acc']
val_loss = history_dict['val_loss']
val_accuracy = history_dict['val_acc']

# Draw loss
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

# Draw acc
plt.figure()
plt.plot(range(epochs), train_accuracy, label='train_accuracy')
plt.plot(range(epochs), val_accuracy, label='val_acc')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')

# Display
plt.show()

print('Task finished')
