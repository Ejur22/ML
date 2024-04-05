
#import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.python import keras
from keras.datasets import cifar10  # набор 60К данных 10 классов
from keras.models import Sequential  # класс для создания слоев
from keras.layers import Flatten, Dense  # классы для создания полносвязных слоёв
from keras.utils import to_categorical  # преобразует вектор в матрицу двоичных классов


def create_model():
    (x_train, y_train), (x_val, y_val) = cifar10.load_data()
    x_train = x_train / 255
    x_val = x_val / 255

    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)

    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(1000, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
    model.save('my_model1000.keras')


def main():
    st.title('Классификатор на основе Cifar10 ')
    st.write('Загрузи картинку')

    file = st.file_uploader('Загрузи .jpg или .png', type=['jpg', 'png'])
    if not file:
        st.text('Это не .jpg или .png')
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        resized = image.resize((32, 32))
        img_array = np.array(resized) / 255  # нормализуем значение пикселей
        img_array = img_array.reshape((1, 32, 32, 3))  # изображений, размеры, каналов

        model = tf.keras.models.load_model('my_model2.keras')
        predictions = model.predict(img_array)

        classes = [
            'самолёт', 'автомобиль', 'птица', 'кошка', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик'
        ]
        fig, ax = plt.subplots()
        y_pos = np.arange(len(classes))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes)
        ax.invert_yaxis()
        ax.set_xlabel('Вероятность')
        ax.set_title('Что это')

        st.pyplot(fig)  # встраивание диаграммы в веб-приложение


if __name__ == '__main__':
    #create_model()
    main()
