import logging

from utils import *
# import numpy as np
# import tensorflow as tf


# Definicion de variables globales
MODELO = 0
COMPUTED_OPTION = 0
SAVE_RESULTS = True
LOW_DATA = True
SPLIT_INTO_TWO_DAYS = False
np.random.seed(42)
tf.random.set_seed(42)
SPLIT = 8
HORIZON = WINDOW_SIZE = None

"""
Analisis de parametros de entrada
"""
if COMPUTED_OPTION > 2:
    raise ValueError("Las opciones de computo son 0 ( minuto ) , 1 ( hora ) y 2 ( dia )")
if MODELO > 2:
    raise ValueError("Los modelos disponibles son 0 ( agregado ) , 1 ( individual ) y 2 ( matricial )")
if COMPUTED_OPTION == 2:
    logging.warning("La predicción en días no esta en uso")


DIR_NAMES = get_dirs_by_model(modelo=MODELO)
#Leemos los datos de los participantes
dataX, dataY = read_data(low_data=LOW_DATA,computed_option=COMPUTED_OPTION,dir_with_data=DIR_NAMES["read"])

X_train,y_train,X_validation,y_validation,X_test,y_test = train_test_validation_split(dataX,dataY,
                                                                                      get_split_time_definiton(num_split=SPLIT))

X_train, y_train, X_validation, y_validation, X_test, y_test = data_splits_preprocessing(
    X_train, y_train, X_validation, y_validation, X_test, y_test,computed_option=COMPUTED_OPTION)

print("Examples for training\n","X:",X_train.shape,"y:",y_train.shape)
print("Examples for validation\n","X:",X_validation.shape,"y:",y_validation.shape)
print("Examples for test\n","X:",X_test.shape,"y:",y_test.shape)

if MODELO == 0:
    HORIZON = y_test.shape[1]
    WINDOW_SIZE = X_test.shape[1]

# Creamos el modelo LSTM
model_LSTM = create_LSTM_model(prediction_horizion=HORIZON,window_size=WINDOW_SIZE,computed_option=COMPUTED_OPTION)

# Lo compilamos y entrenamos usando early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    mode="auto"
)
model_LSTM.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(),
             metrics=["mae"])

hist = model_LSTM.fit(X_train,
            y_train,
            epochs=100,
            verbose=1,
            batch_size=256,
            validation_data=(X_validation, y_validation),
               callbacks=[early_stopping])


predictions = make_preds(model=model_LSTM, input_data=X_test)

info_results = generate_results(
            model=MODELO,
            y_test=y_test,
            predictions=predictions,
            computed_option=COMPUTED_OPTION,
            X_test=X_test,
            horizon=HORIZON,
            model_LSTM=model_LSTM)


