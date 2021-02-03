import os
from os import listdir
from PIL import Image

os.chdir('C:\\Users\\nemes\\OneDrive\\Desktop\\BikeVsCar\\Validation\\MotorBike')

for filename in listdir('C:\\Users\\nemes\\OneDrive\\Desktop\\BikeVsCar\\Validation\\MotorBike'):
    if filename.endswith('.jpg'):
        try:
            img = Image.open('./' + filename)
            img.verify()
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)  # print out the names of corrupt files
            os.remove(os.path.join('C:\\Users\\nemes\\OneDrive\\Desktop\\BikeVsCar\\Training\\Car', filename))


