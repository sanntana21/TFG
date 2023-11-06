from utils import *

def make_mets_predictions(
        *,
        MODELO: int = None,
        COMPUTED_OPTION: int = None,
        SAVE_RESULTS: bool = None,
        LOW_DATA: bool = None,
        SPLIT: int = None,
        SHOW_RESULTS: bool = None,
        seed: int = 42
):
    """
    GENERA LOS RESULTADOS PARA UN DETERMINADO MODELO Y SPLIT SIGUIENDO TODOS LOS PASOS DEL APRENDIZAJE:
        1. PREPROCESAMIENTO DE DATOS
        2. ENTRENAMIENTO DEL MODELO
        3. TESTEO DEL MODELO
    LOS PARÁMETROS DE ENTRADA INDICAN:

    @:param MODELO: Modelo a utilizar ->
            0 : AGGREGATED DATA
            1 : INDIVIDUAL DATA
            2 : MATRIX DATA
    @:param COMPUTED_OPTION: Horizonte de predicción utilizada ->
            0 : HORIZON = MINUTE
            1 : HORIZON = HOUR
            2 : HORIZON = DAY ( deprecated )
    @:param LOW_DATA : Indica si trabajar con todos los datos, o la opción reducida de los mismos
    @:param SPLIT: División de folds utilizada para obtener los resultados
    @:param SAVE_RESUTS : indica si guardar o no los resultados. En caso de false se presentan por pantalla
    @:param SHOW_RESULTS: indica si mostrar los datos por pantalla, específicamente las gráficas que se guardarán
    en formato html.

    @:param SEED: semilla a utilizar para obtener los resultados

    @:return Diccionario con una entrada "data" para los resultados calculados y "errors" en el caso de que ocurra un error
    """
    results = {"data": {}, "errors": []}
    try:
        np.random.seed(seed)
        tf.random.set_seed(seed)
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
        # Leemos los datos de los participantes
        dataX, dataY = read_data(low_data=LOW_DATA, computed_option=COMPUTED_OPTION, dir_with_data=DIR_NAMES["read"])
        NUMBER_OF_PARTICIPANTS = dataX.shape[0]
        """
        Generación de los conjuntos a utilizar
        """
        X_train, y_train, X_validation, y_validation, X_test, y_test = train_test_validation_split(dataX, dataY,
                                                                                                   get_split_time_definiton(
                                                                                                       num_split=SPLIT))

        X_train, y_train, X_validation, y_validation, X_test, y_test = data_splits_preprocessing(
            X_train, y_train, X_validation, y_validation, X_test, y_test, model=MODELO)

        print("Examples for training\n", "X:", X_train.shape, "y:", y_train.shape)
        print("Examples for validation\n", "X:", X_validation.shape, "y:", y_validation.shape)
        print("Examples for test\n", "X:", X_test.shape, "y:", y_test.shape)

        if MODELO < 2:
            HORIZON = y_test.shape[1]
            WINDOW_SIZE = X_test.shape[1]
        else:
            HORIZON = y_test.shape[2]
            WINDOW_SIZE = X_train.shape[2]

        # Creamos el modelo LSTM
        model_LSTM = create_LSTM_model(prediction_horizion=HORIZON, window_size=WINDOW_SIZE, model=MODELO,
                                       number_of_participants=NUMBER_OF_PARTICIPANTS)

        model_LSTM.summary()

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

        #Entrenado el modelo realizamos las predicciones y obtenemos los resultados
        predictions = make_preds(model=model_LSTM, input_data=X_test)

        info_results = generate_results(
            model=MODELO,
            y_test=y_test,
            predictions=predictions,
            computed_option=COMPUTED_OPTION,
            X_test=X_test,
            horizon=HORIZON,
            model_LSTM=model_LSTM,
            window_size=WINDOW_SIZE,
            show=SHOW_RESULTS)

        info_results["hist"] = hist

        #Guardamos los resultados si así se indica
        if SAVE_RESULTS:
            save_results(results_dir_to_save_results=DIR_NAMES["save"], info_results=info_results,
                         computed_option=COMPUTED_OPTION, low_data=LOW_DATA,
                         split=SPLIT)

        results["data"] = info_results
    except BaseException as e:
        logging.error(f"Error al generar los resultados para modelo {MODELO} y opcion {COMPUTED_OPTION}")
        results["errors"].append(e)

    return results


if __name__ == "__main__":
    logging.warning("Generando predicciones de todas las combinaciones")
    for MODELO in [0,1,2]:
        for COMPUTER_OPTION in [0, 1]:
            for SPLIT in [0,1,2,3,4,5,6,7,8]:
                make_mets_predictions(MODELO=MODELO,
                                      COMPUTED_OPTION=COMPUTER_OPTION,
                                      SAVE_RESULTS=True,
                                      LOW_DATA=True,
                                      SPLIT=SPLIT,
                                      SHOW_RESULTS=False,
                                      seed=42)

