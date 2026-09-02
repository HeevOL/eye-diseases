import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, mixed_precision
import os
import matplotlib.pyplot as plt

def configurar_gpu():
    """Configura a GPU para uso eficiente e ativa treinamento com precisão mista"""
    dispositivos_gpu = tf.config.list_physical_devices('GPU')

    if dispositivos_gpu:
        try:
            for dispositivo in dispositivos_gpu:
                tf.config.experimental.set_memory_growth(dispositivo, True)
            print(f"GPU configurada: {len(dispositivos_gpu)} placa(s) encontrada(s)")

            mixed_precision.set_global_policy('mixed_float16')
            print("Precisão mista ativada (mixed_float16)")
        except RuntimeError as erro:
            print(f"Aviso ao configurar GPU: {erro}")
    else:
        print("Nenhuma GPU encontrada. Treinamento vai rodar na CPU.")

IMG_WIDTH, IMG_HEIGHT = 224, 224

BATCH_SIZE = 16
EPOCHS = 20
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'eye_disease_model.h5'

def create_data_generators():
    """Cria os geradores de dados para treino, validação e teste"""

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

def create_cnn_model(num_classes):
    """Cria modelo CNN personalizado"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model():
    """Função principal para treinar o modelo"""

    configurar_gpu()

    print("Criando geradores de dados...")
    train_generator, validation_generator = create_data_generators()

    print(f"Classes detectadas: {train_generator.class_indices}")
    print(f"Número de classes: {train_generator.num_classes}")

    print("Criando modelo CNN...")
    model = create_cnn_model(train_generator.num_classes)

    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    print("Iniciando treinamento...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint]
    )

    model.save('eye_disease_model_final.h5')

    class_indices = train_generator.class_indices
    with open('class_indices.txt', 'w') as f:
        for class_name, index in class_indices.items():
            f.write(f"{class_name}:{index}\n")

    print(f"Modelo salvo como: {MODEL_SAVE_PATH}")
    print(f"Mapeamento de classes salvo como: class_indices.txt")

    plot_training_history(history)

    return model, class_indices

def plot_training_history(history):
    """Plota o histórico de treinamento"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    train_model()