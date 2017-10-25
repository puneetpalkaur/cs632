
http://pillow.readthedocs.io/en/3.4.x/installation.html
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

TRAIN_PATH = sys.argv[1]
VAL_PATH = sys.argv[2]

CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0

# Load extracted files
def load(npy_file):
  data = np.load(npy_file).item()
  return data['images'], data['labels']

train_images, train_labels = load(TRAIN_PATH)
val_images, val_labels = load(VAL_PATH)
print (val_images.shape)

batch_size = 32
# two num classes for cat and dog
num_classes = 2
epochs = 45
data_augmentation = True
num_predictions = 20

# Convert class vectors to binary class matrices.
train_labels = to_categorical(train_labels, num_classes)
val_labels = to_categorical(val_labels, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=train_images.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
train_images = train_images.astype('float32')
val_images = val_images.astype('float32')
train_images /= 255
val_images /= 255

model.fit(train_images, train_labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(val_images, val_labels),
              shuffle=True)
# Save model 
model_path = 'model_1.h5'
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model
scores = model.evaluate(val_images, val_labels, verbose=1)
print(scores)
