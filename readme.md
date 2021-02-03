**Introducere**

Am creat un model pentru clasificarea binara intre imagini care
reprezinta fie o motocicleta fie o masina.

Modelul este creat utilizand framework-urile tensorflow, keras, numpy.

In ideea in care parcarile utilizeaza tot mai des bariere pentru
tarifarea locurilor de parcare, acest model ar putea fi folosit
pentru clasificarea vehiculelor.

In acest fel se realizeaza diferentierea tarifelor in cazul
motocicletelor fata de masini, avand in vedere spatiul ocupat.

**Studiu bibliografic**

Un studiu similar a fost realizat de catre Bhanu Yerra (2019, Car Image Classification Using Features Extracted from Pre-trained Neural Networks) 
prin care s-a urmarit clasificarea a 5 clase de masini din 16.000 imagini (train + test).
Cele 5 clase de masini au fost create in urma consolidarii setului de date, prin gruparea imaginilor in functie de tipul de masina.
Primul pas realizat in construirea modelului a fost reducerea dimensiunii imaginilor
(224 x 224). Structura datelor a fost 80% train si 20% test.
Autorul a utilizat cateva dintre cele mai importante metode de clasificare precum:
Regresie logistica, random forest, XGBoost.
Pentru evaluarea performantei modelelor s-a utilizat in principal valoarea "Accuracy".
In urma evaluarii modelelor au observat ca modelele bazate pe regresie logistica performeaza
cel mai bine cu accuracy de 81% iar cele bazate pe Random Forest au avut de asemenea valori
bune de 76%. In mod similar s-a procedat si de catre alti doi autori, obtinand rezultate similare (intre 70-90%).
Sursa: https://towardsdatascience.com/classifying-car-images-using-features-extracted-from-pre-trained-neural-networks-39692e445a14

**Datele utilizate**

Datele reprezinta imagini cu motociclete si imagini cu masini,
acestea fiind preluate de pe site-ul Kaggle.
Imagini motociclete: https://www.kaggle.com/phucbb/motorbike-zaloai
Imagini masini: https://www.kaggle.com/occultainsights/honda-cars-over-11k-labeled-images

**Construirea modelului**
- prin models.Sequential se vor adauga straturi cu un singur input si un singur output
- s-au adaugat 5 layere convolutionale
- rezultatul cel mai bun s-a obtinut prin utilizarea unui numar de 10-15 epochs, datorita numerului mic de imagini si pentru a evita overfitting
- in urma analizarii studiilor pe aceasta tema, am decis incercarea functiei de activare 'tanh' precum si 'sigmoid', fiind cele mai utilizate pentru clasificari binare


**Rezultate**
a) Utilizand activation function 'tanh'

8/8 [==============================] - 104s 11s/step - loss: 0.7652 - accuracy: 0.4772 - val_loss: 0.7553 - val_accuracy: 0.5508
Epoch 2/10
8/8 [==============================] - 56s 7s/step - loss: 0.6831 - accuracy: 0.5827 - val_loss: 0.7547 - val_accuracy: 0.5586
Epoch 3/10
8/8 [==============================] - 51s 6s/step - loss: 0.8169 - accuracy: 0.4875 - val_loss: 0.8016 - val_accuracy: 0.4766
Epoch 4/10
8/8 [==============================] - 57s 7s/step - loss: 0.7180 - accuracy: 0.5231 - val_loss: 0.7503 - val_accuracy: 0.5039
Epoch 5/10
8/8 [==============================] - 56s 7s/step - loss: 0.6334 - accuracy: 0.6079 - val_loss: 0.5163 - val_accuracy: 0.8477
Epoch 6/10
8/8 [==============================] - 53s 7s/step - loss: 0.4899 - accuracy: 0.8196 - val_loss: 0.4154 - val_accuracy: 0.9062
Epoch 7/10
8/8 [==============================] - 51s 6s/step - loss: 0.4821 - accuracy: 0.8225 - val_loss: 0.4815 - val_accuracy: 0.7305
Epoch 8/10
8/8 [==============================] - 52s 6s/step - loss: 0.4836 - accuracy: 0.7999 - val_loss: 0.3988 - val_accuracy: 0.9180
Epoch 9/10
8/8 [==============================] - 45s 5s/step - loss: 0.4835 - accuracy: 0.8182 - val_loss: 0.4198 - val_accuracy: 0.8711
Epoch 10/10
8/8 [==============================] - 45s 6s/step - loss: 0.4366 - accuracy: 0.8665 - val_loss: 0.4007 - val_accuracy: 0.9062


b) Utilizand activation function 'sigmoid'

8/8 [==============================] - 90s 10s/step - loss: 0.9665 - accuracy: 0.5213 - val_loss: 0.6665 - val_accuracy: 0.5391
Epoch 2/10
8/8 [==============================] - 64s 8s/step - loss: 0.6813 - accuracy: 0.5214 - val_loss: 0.6473 - val_accuracy: 0.6133
Epoch 3/10
8/8 [==============================] - 60s 7s/step - loss: 0.7071 - accuracy: 0.6346 - val_loss: 0.5816 - val_accuracy: 0.7969
Epoch 4/10
8/8 [==============================] - 55s 7s/step - loss: 0.6509 - accuracy: 0.6243 - val_loss: 0.6244 - val_accuracy: 0.6016
Epoch 5/10
8/8 [==============================] - 56s 7s/step - loss: 0.6136 - accuracy: 0.6158 - val_loss: 0.9853 - val_accuracy: 0.5820
Epoch 6/10
8/8 [==============================] - 50s 6s/step - loss: 0.7163 - accuracy: 0.6212 - val_loss: 0.6044 - val_accuracy: 0.6445
Epoch 7/10
8/8 [==============================] - 52s 6s/step - loss: 0.5462 - accuracy: 0.7370 - val_loss: 0.8254 - val_accuracy: 0.6758
Epoch 8/10
8/8 [==============================] - 53s 7s/step - loss: 0.7153 - accuracy: 0.6934 - val_loss: 0.4847 - val_accuracy: 0.7617
Epoch 9/10
8/8 [==============================] - 58s 7s/step - loss: 0.4425 - accuracy: 0.7979 - val_loss: 0.3285 - val_accuracy: 0.8594
Epoch 10/10
8/8 [==============================] - 58s 7s/step - loss: 0.4167 - accuracy: 0.8378 - val_loss: 0.2939 - val_accuracy: 0.8906

Dupa cum se poate observa, cele doua functii obtin rezultate similare, 89% respectiv 90%, 'tanh' obtinand totusi un rezultat superior.
Ca si loss function, s-a utilizat Binary Crossentropy, fiind cea mai utilizata pentru clasificari binare.

**Concluzii**

Plecand de la ideea de baza, de a crea un model care sa diferentieze intre doua tipuri de vehicule: masini si motociclete, 
putem observa ca rezultatele obtinute prezinta un nivel ridicat de acuratete.
In testele realizate de catre noi, modelul a reusit sa diferentieze corect, dintr-un total de 80 de imagini, 30 de motociclete si 50 de masini.

**Surse bibliografice**

https://towardsdatascience.com/classifying-car-images-using-features-extracted-from-pre-trained-neural-networks-39692e445a14

https://www.kaggle.com/occultainsights/honda-cars-over-11k-labeled-images

https://www.kaggle.com/phucbb/motorbike-zaloai

https://github.com/spectrico/car-make-model-classifier-yolo3-python

https://github.com/kaamka/cars-classification-deep-learning
