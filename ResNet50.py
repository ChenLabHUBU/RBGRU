import tensorflow as tf
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
train_dir = r"/home/gump/data/flowers/train"
validation_dir = r"/home/gump/data/flowers/val"
BATCH_SIZE = 32
IMG_SIZE = (224,224)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(validation_dir,shuffle=False,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
class_names = train_dataset.class_names
class_len = len(class_names)

def Conv_Block(input_layer,filters,strides):
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=(1,1),strides=(1, 1),padding="valid",use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation="relu")(x)
    
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=(3,3),strides=strides,padding="same",use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation="relu")(x)
    
    x = tf.keras.layers.Conv2D(filters=4*filters,kernel_size=(1,1),strides=(1, 1),padding="valid",use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    shortcut = tf.keras.layers.Conv2D(filters=4*filters,kernel_size=(1,1),strides=strides,padding="same",use_bias=False)(input_layer)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = tf.keras.layers.Add()([x,shortcut])
    x = tf.keras.layers.Activation(activation="relu")(x)
    
    return x

def Identity_Block(input_layer,filters):
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=(1,1),strides=(1, 1),padding="valid",use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation="relu")(x)
    
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=(3,3),strides=(1, 1),padding="same",use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation="relu")(x)
    
    x = tf.keras.layers.Conv2D(filters=4*filters,kernel_size=(1,1),strides=(1, 1),padding="valid",use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Add()([x,input_layer])
    x = tf.keras.layers.Activation(activation="relu")(x)
    
    return x

input_layer = tf.keras.layers.Input(shape=(224,224,3))
#conv1
x = tf.keras.layers.Conv2D(filters=64,kernel_size=(7,7),strides=(2,2),padding="same",use_bias=False)(input_layer)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(activation="relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(x)
#conv2
x = Conv_Block(input_layer=x,filters=64,strides=(1,1))
for _ in range(2):
    x = Identity_Block(input_layer=x,filters=64)
#conv3
x = Conv_Block(input_layer=x,filters=128,strides=(2,2))
for _ in range(3):
    x = Identity_Block(input_layer=x,filters=128)
#cov4
x = Conv_Block(input_layer=x,filters=256,strides=(2,2))
for _ in range(5):
    x = Identity_Block(input_layer=x,filters=256)
#cov5
x = Conv_Block(input_layer=x,filters=512,strides=(2,2))
for _ in range(2):
    x = Identity_Block(input_layer=x,filters=512)
#global_avg_pool+fc
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
output = tf.keras.layers.Dense(units=class_len,activation="softmax")(x)
model = tf.keras.Model(input_layer,output)

with open(r"./resnet50_class_names.json", 'w', encoding='utf-8') as f:
    json.dump(class_names, f)
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
              ,loss=tf.keras.losses.SparseCategoricalCrossentropy()
              ,metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

checkpoint_path = r"./models/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,save_freq="epoch")


history = model.fit(train_dataset,epochs=100,validation_data=validation_dataset,callbacks=[cp_callback])
pd.DataFrame(history.history).to_csv(r'./resnet50.csv', index=False)
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('ResNet50 Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('ResNet50 Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig(r"./resnet50.png")