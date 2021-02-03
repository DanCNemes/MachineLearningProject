import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

# Director training car
train_car_dir = "C:\\Users\\nemes\\OneDrive\\Desktop\\BikeVsCar\\Training\\Car"

# Director training motorbike
train_motorbike_dir = "C:\\Users\\nemes\\OneDrive\\Desktop\\BikeVsCar\\Training\\MotorBike"

# Director validare car
validation_car_dir = "C:\\Users\\nemes\\OneDrive\\Desktop\\BikeVsCar\\Validation\\Car"

# Director validare motorbike
validation_motorbike_dir = "C:\\Users\\nemes\\OneDrive\\Desktop\\BikeVsCar\\Validation\\MotorBike"

train_car_names = os.listdir(train_car_dir)
print(train_car_names[:10])

train_motorbike_names = os.listdir(train_motorbike_dir)
print(train_motorbike_names[:10])

validation_car_names = os.listdir(validation_car_dir)
print(validation_car_names[:10])

validation_motorbike_names = os.listdir(validation_motorbike_dir)
print(validation_motorbike_names[:10])


print('total training car images:', len(os.listdir(train_car_dir)))
print('total training motorbike images:', len(os.listdir(train_motorbike_dir)))
print('total validation car images:', len(os.listdir(validation_car_dir)))
print('total validation motorbike images:', len(os.listdir(validation_motorbike_dir)))


nrows = 4
ncols = 4


pic_index = 0


fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_car_pix = [os.path.join(train_car_dir, fname)
                for fname in train_car_names[pic_index-8:pic_index]]
next_motorbike_pix = [os.path.join(train_motorbike_dir, fname)
                for fname in train_motorbike_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_car_pix+next_motorbike_pix):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

model = tf.keras.models.Sequential([
    # Convolutia 1
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Convolutia 2
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Convolutia 3
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Convolutia 4
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Convolutia 5
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    # strat ascuns cu 512 neuroni
    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='tanh')
])

model.summary()

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(loss=loss,
              optimizer='rmsprop',
              metrics=['accuracy'])



train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)


train_generator = train_datagen.flow_from_directory(
        "C:\\Users\\nemes\\OneDrive\\Desktop\\BikeVsCar\\Training",
        target_size=(300, 300),
        batch_size=128,

        class_mode='binary')


validation_generator = validation_datagen.flow_from_directory(
        "C:\\Users\\nemes\\OneDrive\\Desktop\\BikeVsCar\\Validation",
        target_size=(300, 300),
        batch_size=32,

        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=8,
      epochs=10,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)