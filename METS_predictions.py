import json
import logging
import sys

from utils import *


# import numpy as np
# import tensorflow as tf

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
    results = {"data": {}, "errors": []}
    try:
        """
        MODELO ->
                0 : AGGREGATED DATA
                1 : INDIVIDUAL DATA
                2 : MATRIX DATA
        COMPUTED_OPTION ->
                0 : HORIZON = MINUTE
                1 : HORIZON = HOUR
                2 : HORIZON = DAY ( deprecated )
        """

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
        model_LSTM = create_LSTM_model(prediction_horizion=HORIZON, window_size=WINDOW_SIZE, model=MODELO)

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
    sys.argv.append({"modelo":0,"save_results":True})
    print(sys.argv)
    if not isinstance(sys.argv[1], dict):
        if sys.argv[1] == "all":
            logging.warning("Generando predicciones de todas las combinaciones")
            for MODELO in [1, 2]:
                for COMPUTER_OPTION in [0, 1]:
                    for SPLIT in [0,1,2,3,4,5,6,7,8]:
                        make_mets_predictions(MODELO=MODELO,
                                              COMPUTED_OPTION=COMPUTER_OPTION,
                                              SAVE_RESULTS=True,
                                              LOW_DATA=True,
                                              SPLIT=SPLIT,
                                              SHOW_RESULTS=False,
                                              seed=42)
        else:
            logging.error("Los parametros de entrada no han sido indicados correctamente")
    else:
        body = sys.argv[1]
        default = {
            "modelo": 0, "computed_option": 0, "save_results": False, "split": 0, "show_results": False, "seed": 42,
            "low_data": True
        }
        for param in ["modelo", "computed_option", "low_data", "save_results", "split", "show_results", "seed"]:
            if not body.get(param):
                body[param] = default[param]

        results = make_mets_predictions(
            MODELO=body["modelo"],
            COMPUTED_OPTION=body["computed_option"],
            SAVE_RESULTS=body["save_results"],
            LOW_DATA=body["low_data"],
            SPLIT=body["split"],
            SHOW_RESULTS=body["show_results"],
            seed=body["seed"]
        )

        if len(results["errors"]) < 1:
            info_results = results["data"]
            text_results = dict(
                POINT_TO_POINT_MAE=info_results["POINT_TO_POINT_MAE"],
                POINT_TO_POINT_MSE=info_results["POINT_TO_POINT_MSE"],
                POBLATIONAL_MAE=info_results["POBLATIONAL_MAE"],
                POBLATIONAL_MSE=info_results["POBLATIONAL_MSE"],
                BEST=info_results["BEST_MEDIUM_WORST_MAE"][2],
                BEST_OLS=info_results["BEST_MEDIUM_WORST_VALUE_DISPERSION"][2],
                MEDIUM=info_results["BEST_MEDIUM_WORST_MAE"][1],
                MEDIUM_OLS=info_results["BEST_MEDIUM_WORST_VALUE_DISPERSION"][1],
                WORST=info_results["BEST_MEDIUM_WORST_MAE"][0],
                WORST_OLS=info_results["BEST_MEDIUM_WORST_VALUE_DISPERSION"][0],
                TWO_DAYS_RESULTS_REAL=info_results["TWO_DAYS_RESULTS_REAL"],
                TWO_DAYS_RESULTS_PREDICTED=info_results["TWO_DAYS_RESULTS_PREDICTED"],
                TWO_DAYS_OLS=info_results["TWO_DAYS_RESULTS_DISPERSION_VALUE"],
                TRANSFORMED_PREDICITION_OLS=info_results["TRANSFORMED_PREDICITION_DISPERSION_VALUE"],
                MEAN_Y=info_results["MEAN_Y"],
                POBLATIONAL_MEAN=info_results["POBLATIONAL_MEAN"],
                HIST=info_results["hist"].history
            )
            text_results = {k: str(v) for k, v in text_results.items()}

            print(json.dumps(text_results, indent=2))
