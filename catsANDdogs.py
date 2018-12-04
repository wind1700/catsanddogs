# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 08:39:02 2018

@author: Administrator
"""

from keras import layers
from keras import models


model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',
                        input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


#model.summary()
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
 rotation_range=40,
 width_shift_range=0.2,
 height_shift_range=0.2,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True,
 fill_mode='nearest')

train_datagen=ImageDataGenerator(rescale=1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

#train_cats_generator=train_datagen.flow_from_directory(
      #  "F:\\kaggle数据集\\all(cats and dogs)\\train\\train_cats",
       # target_size=(150,150),
       # batch_size=20,
       # class_mode='binary')

#train_dogs_generator=train_datagen.flow_from_directory(
        #"F:\\kaggle数据集\\all(cats and dogs)\\train\\train_dogs",
       # target_size=(150,150),
       # batch_size=20,
       # class_mode='binary')

train_generator=train_datagen.flow_from_directory(
        "./small",
       target_size=(150,150),
       batch_size=20,
       class_mode='binary')





validation_generator = test_datagen.flow_from_directory(
 "./test1",
 target_size=(150, 150),
 batch_size=20,
 class_mode='binary')




history = model.fit_generator(
        train_generator,
         steps_per_epoch=100,
         epochs=30,
         validation_data=validation_generator,
         validation_steps=50)


model.save('cats_and_dogs_small_1.h5')



import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()




#from keras.models import load_model

#model= load_model('cats_and_dogs_small_1.h5')














