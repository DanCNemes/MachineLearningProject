import os
from keras.preprocessing import image
import numpy as np
import keras
from os import listdir
import shutil

# Load model
model = keras.models.load_model('C:\\Users\\nemes\\PycharmProjects\\TensorFlowProject')

for filename in listdir('C:\\Users\\nemes\\OneDrive\\Desktop\\SADE_Py\\Vehicule'):
    file_path = os.path.join('C:\\Users\\nemes\\OneDrive\\Desktop\\SADE_Py\\Vehicule', filename)
    img = image.load_img(file_path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    if classes[0] > 0.5:
        shutil.move(file_path, os.path.join('C:\\Users\\nemes\\OneDrive\\Desktop\\SADE_Py\\Classification_output\\Motorcycle', filename))
        print(filename + " is a motorbike with a probability of: ", classes[0])
    else:
        shutil.move(file_path, os.path.join('C:\\Users\\nemes\\OneDrive\\Desktop\\SADE_Py\\Classification_output\\Car', filename))
        print(filename + " is a car with a probability of:", 1 - classes[0])