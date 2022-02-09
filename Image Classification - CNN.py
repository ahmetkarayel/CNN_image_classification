import tensorflow as tf
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model = models.Sequential()

# Convolutional (Evrişim) Katmanları
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Dropout Katmanları -> Overfitting önlemek için
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid')) # food or non-food




model.compile(loss="binary_crossentropy",
              optimizer = optimizers.RMSprop(learning_rate=1e-4),
              metrics = ["acc"])

# Oluşturduğumuz modelin özeti
model.summary()

# Resimlerimizin bulunduğu dizinler
train_dir = 'food_vs_non-food/Train'
validation_dir = 'food_vs_non-food/Validation'
test_dir = 'food_vs_non-food/Test'


# Data Augmentation -> overfitting önlemek için
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (200, 200),
    batch_size = 150,
    class_mode = "binary"
)


# Validation datamız için data augmentation işlemine gerek yok.
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size= (200, 200),
    batch_size= 30,
    class_mode = "binary"
)



# Oluşturduğumuz modeli eğitiyoruz.
history = model.fit_generator(
    train_generator,
    steps_per_epoch = 10, # batch_size:150 --> 150 x 10 = 1500 resim
    epochs = 20,
    validation_data = validation_generator,
    validation_steps = 10 # 
    
)

model.save("food_non-food.h5")


# Train - Validation "Acc" - "Loss" grafiklerimiz
acc = history.history["acc"]
val_acc = history.history["val_acc"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, "r", label="Train Acc")
plt.plot(epochs, val_acc, "b", label="Validation Acc")
plt.title("Train-Validation Acc")
plt.legend()
plt.show()

plt.plot(epochs, loss, "r", label="Train Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Train-Validation Loss")
plt.legend()
plt.show()

# Modelin görmediği (test) datası üzerinde başarımız.
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)