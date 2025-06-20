import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import os
import json

def create_transfer_learning_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model
#Prepare Data
def prepare_data(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='training')

    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical')

    return train_generator, val_generator

def train_model(model, train_generator, val_generator, epochs=10):
    model.compile(optimizer=Adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_generator,
              validation_data=val_generator,
              epochs=epochs,
              verbose=1)
    
    return model, train_generator.class_indices

#Mainmethod
def main():
    train_dir = 'dataset/train'
    val_dir = 'dataset/validation'
    num_classes = len(os.listdir(train_dir))

    model = create_transfer_learning_model(num_classes)
    train_gen, val_gen = prepare_data(train_dir, val_dir)
    model, class_indices = train_model(model, train_gen, val_gen)

    model.save('transfer_learning_model.h5')

    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)

if __name__ == "__main__":
    main()
