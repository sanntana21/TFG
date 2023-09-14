import gzip
import pandas as pd
import statsmodels
from statistics import mean, median
import numpy as np
import pickle
import os
import json
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import logging
import plotly.express as px
import tensorflow as tf
import gzip

def set_output_precision(decimals):
    """
    format the output of the all the data structures
    with an specific number of decimals
    """
    np.set_printoptions(precision=decimals)
    into='{'+':.{}f'.format(decimals)+'}'
    pd.options.display.float_format = into.format
    pass

def plot_ts(df,dfx="Minute",dfy="METS",_title="DF minute x Mets"):
    if not isinstance(df,pd.DataFrame):
        df = pd.DataFrame({'METS': df, 'Minute': range(len(df))})

    plt.figure()
    fig = px.line(df, x = dfx, y = dfy , title = _title)
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                dict(count=1,label="1y",step="year",stepmode="backward"),
                dict(count=2,label="2y",step="year",stepmode="backward"),
                dict(count=3,label="3y",step="year",stepmode="backward"),
                dict(step="all")
            ])
        )

    )
    fig.show()

def plot_predictions_vs_real(predictions, reals):
    df = pd.DataFrame()
    number_of_points = len(predictions)
    df["time"] = range(0,number_of_points)
    df["participant"] = "prediction"
    df["value"] = predictions
    for i in range(0,number_of_points):
        df.loc[number_of_points+i] = [i,"real",reals[i]]

    plt.figure(1)

    fig = px.line(df, x = "time", y = "value" , title = "Predictions vs Reals Time Series" , color = "participant",template='seaborn')
    fig.update_layout(
    plot_bgcolor='white'
    )

    fig.update_xaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey'
    )

    fig.update_yaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey'
    )

    fig2 = fig
    fig2.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                dict(count=1,label="1y",step="year",stepmode="backward"),
                dict(count=2,label="2y",step="year",stepmode="backward"),
                dict(count=3,label="3y",step="year",stepmode="backward"),
                dict(step="all")
            ])
        )

    )


    return fig,fig2

def plot_dispersion_in_predictions(predictions, reals,max_=None,min_=None):
    df = pd.DataFrame()
    number_of_points = len(predictions)
    df["real"] = reals
    df["prediction"] = predictions

    if not max_:
        maximun = max(df["real"].max(),df["prediction"].max())
    if not min_:
        minimun = min(df["real"].min(),df["prediction"].min())

    range_to_plot = [minimun-10,maximun+10]

    plt.figure(1)
    fig = px.scatter(df, x = "real", y = "prediction" , title = "Actual vs Predicted Values",trendline="ols",range_x=range_to_plot,range_y=range_to_plot,trendline_color_override='black',template='seaborn')
    fig.update_layout(
    plot_bgcolor='white'
    )
    fig.update_xaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey'
    )
    fig.update_yaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey'
    )
    parameters = px.get_trendline_results(fig)["px_fit_results"].get(0).params
    return fig,parameters

def get_dirs_by_model(
        *,
        modelo : int = 0,
):
    if modelo == 0:
        dirs = {"read": "Agregado", "save": "Agregado"}
    elif modelo == 1:
        dirs = {"read": "Individual", "save": "Individual"}
    else:
        dirs = {"read": "Individual", "save": "Matrix"}
    return dirs

def read_data(
        *,
        low_data : bool = None,
        computed_option : int = None,
        dir_with_data : str = None
):
    dataX = dataY = None
    try:
        PATH = f"Resources/{dir_with_data}/"

        if low_data:
            PATH += "LowData/"

        documents = ['minuteY','hourY','dayY']
        file = PATH+"minuteX.pkl.gz"
        dataX = np.array(pickle.load(gzip.open(file, 'rb')),np.float32)

        file = PATH+documents[computed_option]+".pkl.gz"
        dataY = np.array(pickle.load(gzip.open(file, 'rb')),np.float32)

        #Para el caso de los datos agregados sumamos los valores de los participantes
        # en cada instante de tiempo
        if "Agregado" == dir_with_data:
            dataX = np.sum(dataX,axis=0,keepdims=True)
            dataY = np.sum(dataY,axis=0,keepdims=True)

    except:
        logging.error("Error al leer los datos")

    return dataX,dataY

def calculate_index(time):
  minute_index = time["day"] * 1440 + time["hour"]*60 + time["minute"]
  return int(minute_index)

def get_split(dataX,dataY,index):
    """
    Devuelve un df con el subconjunto de los días,horas,minutos indicados en index
    """
    start = calculate_index(index[0]["start"])
    end = calculate_index(index[0]["end"])
    X_split = dataX[:,start:end,:]
    y_split = dataY[:,start:end,:]
    if len(index) > 1:
        for i in range(1,len(index)):
            start = calculate_index(index[i]["start"])
            end = calculate_index(index[i]["end"])
            X_split = np.concatenate((X_split,dataX[:,start:end,:]),axis=1)
            y_split = np.concatenate((y_split,dataY[:,start:end,:]),axis=1)
    return X_split,y_split

def train_test_validation_split(dataX,dataY,indexs):
    """
    Realiza una partición de valores train | test | validation
    a partir de una definicion en días | horas | minutos de cada uno de los conjuntos
    """
    X_train,y_train = get_split(dataX,dataY,indexs["train"])
    X_validation,y_validation = get_split(dataX,dataY,indexs["validation"])
    X_test,y_test = get_split(dataX,dataY,indexs["test"])
    return X_train,y_train,X_validation,y_validation,X_test,y_test

def create_LSTM_model(
        *,
        prediction_horizion : int = None,
        window_size: int = None,
        model : int = None
):
    tf.random.set_seed(42)
    # Setup dataset hyperparameters
    HORIZON = prediction_horizion
    WINDOW_SIZE = window_size
    model_LSTM = None

    if model < 2:
        # Let's build an LSTM model with the Functional API
        inputs = tf.keras.layers.Input(shape=(WINDOW_SIZE))
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)  # expand input dimension to be compatible with LSTM
        # print(x.shape)
        # x = layers.LSTM(128, activation="relu", return_sequences=True)(x) # this layer will error if the inputs are not the right shape
        x =tf.keras.layers.LSTM(128, activation="relu")(x)  # using the tanh loss function results in a massive error
        # print(x.shape)
        # Add another optional dense layer (you could add more of these to see if they improve model performance)
        # x = layers.Dense(32, activation="relu")(x)
        output = tf.keras.layers.Dense(HORIZON)(x)
        model_LSTM = tf.keras.Model(inputs=inputs, outputs=output, name="model_5_lstm")

    return model_LSTM

def make_preds(
        *,
        model,
        input_data):
    """
    Realiza predicciones para los datos de entrada y devuelve predicciones 1d
    """
    forecast = model.predict(input_data,verbose=2)
    return tf.squeeze(forecast)


def save_results(
    *,
    results_dir_to_save_results : str = None,
    low_data : bool = True,
    split : int = None,
    computed_option : int = None,
    info_results : dict = None
):
    try:
        file_path = f'Resources/Resultados/{results_dir_to_save_results}/'

        if low_data:
            file_path += "LowData/"

        file_path += "Split" + str(split) + "/"

        if computed_option == 0:
            file_path += "minute/"
        else:
            file_path += "hour/"

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
        isExist = os.path.exists(file_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(file_path)
            logging.warning("Nuevo directorio creado:" + file_path)

        with open(file_path + "/info_resultados.txt", 'w') as f:
            json.dump(json.dumps(str(text_results), indent=4), f, ensure_ascii=False, indent=4)

        # Guardamos las imagenes restantes
        info_results.get("TWO_DAYS_RESULTS_DISPERSION_FIG").write_html(file_path + "/TWO_DAYS_OLS" + ".html")
        info_results.get("BEST_MEDIUM_WORST_FIG")[0].write_html(file_path + "/WORST_MAE" + ".html")
        info_results.get("BEST_MEDIUM_WORST_FIG")[1].write_html(file_path + "/AVERAGE_MAE" + ".html")
        info_results.get("BEST_MEDIUM_WORST_FIG")[2].write_html(file_path + "/BEST_MAE" + ".html")
        info_results.get("BEST_MEDIUM_WORST_FIG_DISPERSION")[0].write_html(file_path + "/WORST_MAE_OLS" + ".html")
        info_results.get("BEST_MEDIUM_WORST_FIG_DISPERSION")[1].write_html(file_path + "/AVERAGE_MAE_OLS" + ".html")
        info_results.get("BEST_MEDIUM_WORST_FIG_DISPERSION")[2].write_html(file_path + "/BEST_MAE_OLS" + ".html")
        info_results.get("TWO_DAYS_RESULTS_DISPERSION_FIG").write_html(file_path + "/TWO_DAYS_FIG" + ".html")
        info_results.get("TRANSFORMED_PREDICITION_FIG").write_html(file_path + "/TRANSFORMED_PREDICTION" + ".html")
        info_results.get("TRANSFORMED_PREDICITION_DISPERSION_FIG").write_html(file_path + "/TRANSFORMED_PREDICTION_OLS" + ".html"),
        info_results.get("TWO_DAYS_RESULTS_FIG_ZOOMED").write_html(file_path + "/TWO_DAYS_RESULTS_FIG" + ".html")
        learning_curves(hist=info_results["hist"],file_path=file_path)
    except:
        logging.error("Error al guardar los resultados")

    return True
def change_shape_by_participant(data):
    original_shape = data.shape
    new_shape = (original_shape[0] * original_shape[1], original_shape[2])
    reshaped_array = data.reshape(new_shape)
    return reshaped_array
def data_splits_preprocessing(
        X_train, y_train, X_validation, y_validation, X_test, y_test,
        model):
    try:
        if model == 0:
            X_train, y_train, X_validation, y_validation, X_test, y_test = [
                np.squeeze(i) for i in [X_train, y_train, X_validation, y_validation, X_test, y_test]
            ]
        elif model == 1:
            X_train, y_train, X_validation, y_validation, X_test, y_test = [i.transpose(1, 0, 2) for i in
                                                                            [X_train, y_train, X_validation,
                                                                             y_validation, X_test, y_test]
                                                                            ]

            X_train, y_train, X_validation, y_validation, X_test, y_test = [change_shape_by_participant(i) for i in
                                                                            [X_train, y_train, X_validation,
                                                                             y_validation, X_test, y_test]
                                                                            ]


    except:
        logging.error("Error al preprocesar los conjuntos")

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def generate_results(
        *,
        model,
        y_test,
        predictions,
        computed_option,
        X_test,
        horizon,
        model_LSTM,
        show,
        window_size
):
    info_results = {}
    if model == 0:
       info_results = generate_results_model_0(y_test=y_test,
            predictions=predictions,
            computed_option=computed_option,
            X_test=X_test,
            horizon=horizon,
            model_LSTM=model_LSTM,show=show)
    elif model == 1:
        info_results = generate_results_model_1(y_test=y_test,
                                                predictions=predictions,
                                                computed_option=computed_option,
                                                X_test=X_test,
                                                horizon=horizon,
                                                window_size=window_size,
                                                model_LSTM=model_LSTM, show=show)

    return info_results
def generate_results_model_0(
        *,
        y_test,
        predictions,
        computed_option,
        X_test,
        horizon,
        model_LSTM,
        show):
    info_results = {}
    info_results["POINT_TO_POINT_MSE"]= mean_squared_error(y_test, predictions)
    info_results["POINT_TO_POINT_MAE"] = mean_absolute_error(y_test, predictions)
    info_results["MEAN_Y"] = np.mean(y_test)
    info_results["POBLATIONAL_MAE"] = mean_absolute_error(np.sum(y_test, axis=1), np.sum(predictions, axis=1))
    info_results["POBLATIONAL_MSE"] = mean_squared_error(np.sum(y_test, axis=1), np.sum(predictions, axis=1))
    info_results["POBLATIONAL_MEAN"] = np.mean(np.sum(y_test, axis=1))
    info_results["BEST_MEDIUM_WORST_MAE"] = [0, 0, 0]
    info_results["BEST_MEDIUM_WORST_FIG"] = [0, 0, 0]
    info_results["BEST_MEDIUM_WORST_FIG_DISPERSION"] = [0, 0, 0]
    info_results["BEST_MEDIUM_WORST_VALUE_DISPERSION"] = [0, 0, 0]

    #Extraemos el mejor,peor y el medio de los valores MAE
    list_of_MAE = [mean_absolute_error(predictions[i], y_test[i]) for i in range(0, len(y_test))]
    list_of_values = sorted(list_of_MAE)
    mean_value = mean(list_of_MAE)
    closest_value = min(list_of_MAE, key=lambda x: abs(x - mean_value))
    index_result = 0
    # Crear un array de índices
    indices = [list_of_MAE.index(list_of_values[-1]),
               list_of_MAE.index(closest_value),
               list_of_MAE.index(list_of_values[0])]
    if computed_option == 0:
        for i in indices:
            info_results["BEST_MEDIUM_WORST_MAE"][index_result] = list_of_MAE[i]
            info_results["BEST_MEDIUM_WORST_FIG"][index_result], _ = plot_predictions_vs_real(predictions[i], y_test[i])
            info_results["BEST_MEDIUM_WORST_FIG_DISPERSION"][index_result], info_results["BEST_MEDIUM_WORST_VALUE_DISPERSION"][
                index_result] = plot_dispersion_in_predictions(predictions[i], y_test[i])
            if show:
                info_results["BEST_MEDIUM_WORST_FIG"][index_result].show()
            index_result += 1
    else:
        for i in indices:
            info_results["BEST_MEDIUM_WORST_MAE"][index_result] = list_of_MAE[i]
            END = 24
            STARTED_MINUTE = 0
            previous = np.ones(shape=(24))
            for j in range(0, 24):
                previous[j] = np.sum(X_test[i, :][60 * j:60 * (j + 1)])
            predictions_to_plot = np.ones(shape=(END + horizon))
            predictions_to_plot[0:END] = previous[:]
            predictions_to_plot[END:] = predictions[i, :]
            y_test_to_plot = np.ones(shape=(END + horizon))
            y_test_to_plot[0:END] = previous[:]
            y_test_to_plot[END:] = y_test[i, :]
            info_results["BEST_MEDIUM_WORST_FIG"][index_result], _ = plot_predictions_vs_real(predictions_to_plot, y_test_to_plot)
            info_results["BEST_MEDIUM_WORST_FIG_DISPERSION"][index_result], info_results["BEST_MEDIUM_WORST_VALUE_DISPERSION"][
                index_result] = plot_dispersion_in_predictions(predictions_to_plot, y_test_to_plot)
            if show:
                info_results["BEST_MEDIUM_WORST_FIG"][index_result].show()
            index_result += 1

    index = 0
    period = X_test[::120, :]
    period_results = make_preds(model=model_LSTM, input_data=period)
    period_results_to_plot = np.array(period_results).reshape(horizon * 23)
    y_test_to_plot = y_test[::120, :].reshape(horizon* 23)
    info_results["TWO_DAYS_RESULTS_FIG"], \
       info_results["TWO_DAYS_RESULTS_FIG_ZOOMED"] = plot_predictions_vs_real(predictions=period_results_to_plot,
                                                               reals=y_test_to_plot)
    info_results["TWO_DAYS_RESULTS_PREDICTED"] = np.sum(period_results_to_plot)
    info_results["TWO_DAYS_RESULTS_REAL"] = np.sum(y_test_to_plot)
    info_results["TWO_DAYS_RESULTS_DISPERSION_FIG"], \
        info_results["TWO_DAYS_RESULTS_DISPERSION_VALUE"] = plot_dispersion_in_predictions(period_results_to_plot, y_test_to_plot)
    if show:
        info_results["TWO_DAYS_RESULTS_FIG"].show()
        info_results["TWO_DAYS_RESULTS_DISPERSION_FIG"].show()

    predictions_transformed = []
    test_transformed = []
    for i in range(0, len(period_results_to_plot), horizon):
        group_sum = np.sum(period_results_to_plot[i:i + horizon])
        predictions_transformed.append(group_sum)
        group_sum = np.sum(y_test_to_plot[i:i + horizon])
        test_transformed.append(group_sum)

    # Convert the transformed list into a numpy array
    predictions_transformed = np.array(predictions_transformed)
    test_transformed = np.array(test_transformed)
    info_results["TRANSFORMED_PREDICITION_FIG"], _ = plot_predictions_vs_real(predictions_transformed, test_transformed)
    info_results["TRANSFORMED_PREDICITION_DISPERSION_FIG"], info_results["TRANSFORMED_PREDICITION_DISPERSION_VALUE"]\
        = plot_dispersion_in_predictions(predictions_transformed, test_transformed)

    return info_results

def generate_results_model_1(
        *,
        y_test,
        predictions,
        computed_option,
        X_test,
        horizon,
        model_LSTM,
        window_size,
        show
):
    info_results = {}
    DATA_BY_PARTICIPANT = int(y_test.shape[0] / 25)
    poblational_prediction = np.ones(shape=(DATA_BY_PARTICIPANT, horizon))
    poblational_y_test = np.ones(shape=(DATA_BY_PARTICIPANT, horizon))
    poblational_X_test = np.ones(shape=(DATA_BY_PARTICIPANT, window_size))
    for i in range(0, DATA_BY_PARTICIPANT):
        poblational_prediction[i, :] = np.sum(np.array(predictions[i::DATA_BY_PARTICIPANT]), axis=0)
        poblational_y_test[i, :] = np.sum(np.array(y_test[i::DATA_BY_PARTICIPANT]), axis=0)
        poblational_X_test[i, :] = np.sum(np.array(X_test[i::DATA_BY_PARTICIPANT]), axis=0)
    info_results["MEAN_Y"] = np.mean(poblational_y_test)
    info_results["POINT_TO_POINT_MAE"] = mean_absolute_error(poblational_y_test, poblational_prediction)
    info_results["POINT_TO_POINT_MSE"] = mean_absolute_error(poblational_y_test, poblational_prediction)

    index_result = 0
    info_results["BEST_MEDIUM_WORST_MAE"]= [0, 0, 0]
    info_results["BEST_MEDIUM_WORST_FIG"] = [0, 0, 0]
    info_results["BEST_MEDIUM_WORST_FIG_DISPERSION"]= [0, 0, 0]
    info_results["BEST_MEDIUM_WORST_VALUE_DISPERSION"] = [0, 0, 0]
    info_results["POBLATIONAL_MAE"]= mean_squared_error(np.sum(poblational_y_test, axis=1), np.sum(poblational_prediction, axis=1))
    info_results["POBLATIONAL_MSE"]= mean_absolute_error(np.sum(poblational_y_test, axis=1), np.sum(poblational_prediction, axis=1))
    info_results["POBLATIONAL_MEAN"] = np.mean(np.sum(poblational_y_test, axis=1))
    list_of_MAE = [mean_absolute_error(poblational_prediction[i], poblational_y_test[i]) for i in
                   range(0, len(poblational_y_test))]
    list_of_values = sorted(list_of_MAE)
    mean_value = mean(list_of_MAE)
    closest_value = min(list_of_MAE, key=lambda x: abs(x - mean_value))
    # Crear un array de índices
    indices = [list_of_MAE.index(list_of_values[-1]),
               list_of_MAE.index(closest_value),
               list_of_MAE.index(list_of_values[0])]
    if computed_option == 0:
        for i in indices:
            info_results["BEST_MEDIUM_WORST_MAE"][index_result] = list_of_MAE[i]
            info_results["BEST_MEDIUM_WORST_FIG"][index_result], _ = plot_predictions_vs_real(predictions[i], y_test[i])
            info_results["BEST_MEDIUM_WORST_FIG_DISPERSION"][index_result], \
                info_results["BEST_MEDIUM_WORST_VALUE_DISPERSION"][index_result] = plot_dispersion_in_predictions(predictions[i],
                                                                                                  y_test[i])
            if show:
                info_results["BEST_MEDIUM_WORST_FIG"][index_result].show()
            index_result += 1
    else:
        for i in indices:
            print(list_of_MAE[i])
            END = 24
            STARTED_MINUTE = 0
            previous = np.ones(shape=(24))
            for j in range(0, 24):
                previous[j] = np.sum(poblational_X_test[i, :][60 * j:60 * (j + 1)])
            predictions_to_plot = np.ones(shape=(END + horizon))
            predictions_to_plot[0:END] = previous[:]
            predictions_to_plot[END:] = poblational_prediction[i, :]
            y_test_to_plot = np.ones(shape=(END + horizon))
            y_test_to_plot[0:END] = previous[:]
            y_test_to_plot[END:] = poblational_y_test[i, :]
            info_results["BEST_MEDIUM_WORST_FIG"][index_result], _ = plot_predictions_vs_real(predictions_to_plot, y_test_to_plot)
            info_results["BEST_MEDIUM_WORST_FIG_DISPERSION"][index_result], info_results["BEST_MEDIUM_WORST_VALUE_DISPERSION"][
                index_result] = plot_dispersion_in_predictions(predictions_to_plot, y_test_to_plot)
            if show:
                info_results["BEST_MEDIUM_WORST_FIG"][index_result].show()
            index_result += 1
            plot_predictions_vs_real(predictions_to_plot, y_test_to_plot)

    period = poblational_X_test[::120, :]
    period_results = make_preds(model=model_LSTM,input_data=period)
    period_results_aux = np.array(period_results)
    period_results_to_plot = np.array(period_results_aux).reshape(horizon* 23)
    y_test_to_plot = poblational_y_test[::120, :].reshape(horizon* 23)
    plot_predictions_vs_real(predictions=period_results_to_plot, reals=y_test_to_plot)
    info_results["TWO_DAYS_RESULTS_FIG"], \
        info_results["TWO_DAYS_RESULTS_FIG_ZOOMED"]= plot_predictions_vs_real(predictions=period_results_to_plot,
                                                               reals=y_test_to_plot)
    info_results["TWO_DAYS_RESULTS_PREDICTED"]= np.sum(period_results_to_plot)
    info_results["TWO_DAYS_RESULTS_REAL"]= np.sum(y_test_to_plot)
    info_results["TWO_DAYS_RESULTS_DISPERSION_FIG"], \
        info_results["TWO_DAYS_RESULTS_DISPERSION_VALUE"]= plot_dispersion_in_predictions(period_results_to_plot, y_test_to_plot)
    if show:
        info_results["TWO_DAYS_RESULTS_FIG"].show()
        info_results["TWO_DAYS_RESULTS_DISPERSION_FIG"].show()
    predictions_transformed = []
    test_transformed = []
    for i in range(0, len(period_results_to_plot), horizon):
        group_sum = np.sum(period_results_to_plot[i:i + horizon])
        predictions_transformed.append(group_sum)
        group_sum = np.sum(y_test_to_plot[i:i + horizon])
        test_transformed.append(group_sum)

    # Convert the transformed list into a numpy array
    predictions_transformed = np.array(predictions_transformed)
    test_transformed = np.array(test_transformed)
    info_results["TRANSFORMED_PREDICITION_FIG"], _ = plot_predictions_vs_real(predictions_transformed, test_transformed)
    info_results["TRANSFORMED_PREDICITION_DISPERSION_FIG"], info_results["TRANSFORMED_PREDICITION_DISPERSION_VALUE"]= plot_dispersion_in_predictions(
        predictions_transformed, test_transformed)
    return info_results

def learning_curves(
        *,
        hist,
        file_path : str = None
):
    fig, loss_ax = plt.subplots()

    loss_ax.plot(hist.history["loss"], "y", label = "train_loss")
    loss_ax.plot(hist.history["val_loss"], "r", label = "val_loss")
    loss_ax.set_ylabel("loss")
    loss_ax.legend(loc = "upper left")
    plt.savefig(file_path+"/curvas_aprendizaje")
    pass
def get_split_time_definiton(
        *,
        num_split : int = None
):
    #Definicion de los splits en dias horas minutos....
    INDEXS = [
        #SPLIT 0
        {"train":[{"start":{"day":2,"hour":0,"minute":0},
                   "end":{"day":21,"hour":22,"minute":0}}
                  ],
         "validation":[{"start":{"day":23,"hour":0,"minute":0},
                        "end":{"day":24,"hour":22,"minute":0}}
                       ],
         "test":[{"start":{"day":26,"hour":0,"minute":0},
                  "end":{"day":27,"hour":22,"minute":0}}
                 ]
         },
        #SPLIT 1
        {
            "train":[{"start":{"day":5,"hour":0,"minute":0},
                      "end":{"day":24,"hour":22,"minute":0}}
                     ],
            "validation":[{"start":{"day":26,"hour":0,"minute":0},
                           "end":{"day":27,"hour":22,"minute":0}}
                          ],
            "test":[{"start":{"day":2,"hour":0,"minute":0},
                     "end":{"day":3,"hour":22,"minute":0}},
                    ]
        },
        # SPLIT 2
        {
            "train":[{"start":{"day":7,"hour":0,"minute":0},
                      "end":{"day":28,"hour":22,"minute":0}}
                     ],
            "validation":[{"start":{"day":2,"hour":0,"minute":0},
                           "end":{"day":3,"hour":22,"minute":0}}
                          ],
            "test":[{"start":{"day":5,"hour":0,"minute":0},
                     "end":{"day":6,"hour":22,"minute":0}},
                    ]
        },
        #SPLIT 3
        {
            "train":[{"start":{"day":10,"hour":0,"minute":0},
                      "end":{"day":28,"hour":22,"minute":0}},
                     {"start":{"day":2,"hour":0,"minute":0},
                      "end":{"day":3,"hour":22,"minute":0}}
                     ],
            "validation":[{"start":{"day":5,"hour":0,"minute":0},
                           "end":{"day":6,"hour":22,"minute":0}}
                          ],
            "test":[{"start":{"day":8,"hour":0,"minute":0},
                     "end":{"day":9,"hour":22,"minute":0}},
                    ]
        },
        #SPLIT 4
        {
            "train":[{"start":{"day":14,"hour":0,"minute":0},
                      "end":{"day":28,"hour":22,"minute":0}},
                     {"start":{"day":2,"hour":0,"minute":0},
                      "end":{"day":6,"hour":22,"minute":0}}
                     ],
            "validation":[{"start":{"day":8,"hour":0,"minute":0},
                           "end":{"day":9,"hour":22,"minute":0}}
                          ],
            "test":[{"start":{"day":11,"hour":0,"minute":0},
                     "end":{"day":12,"hour":22,"minute":0}},
                    ]
        },
        #SPLIT 5
        {
            "train":[{"start":{"day":17,"hour":0,"minute":0},
                      "end":{"day":28,"hour":22,"minute":0}},
                     {"start":{"day":2,"hour":0,"minute":0},
                      "end":{"day":9,"hour":22,"minute":0}}
                     ],
            "validation":[{"start":{"day":11,"hour":0,"minute":0},
                           "end":{"day":12,"hour":22,"minute":0}}
                          ],
            "test":[{"start":{"day":14,"hour":0,"minute":0},
                     "end":{"day":15,"hour":22,"minute":0}},
                    ]
        },
        #SPLIT 6
        {
            "train":[{"start":{"day":20,"hour":0,"minute":0},
                      "end":{"day":28,"hour":22,"minute":0}},
                     {"start":{"day":2,"hour":0,"minute":0},
                      "end":{"day":13,"hour":22,"minute":0}}
                     ],
            "validation":[{"start":{"day":14,"hour":0,"minute":0},
                           "end":{"day":15,"hour":22,"minute":0}}
                          ],
            "test":[{"start":{"day":17,"hour":0,"minute":0},
                     "end":{"day":18,"hour":22,"minute":0}},
                    ]
        },
        # SPLIT 7
        {
            "train":[{"start":{"day":23,"hour":0,"minute":0},
                      "end":{"day":28,"hour":22,"minute":0}},
                     {"start":{"day":2,"hour":0,"minute":0},
                      "end":{"day":15,"hour":22,"minute":0}}
                     ],
            "validation":[{"start":{"day":17,"hour":0,"minute":0},
                           "end":{"day":18,"hour":22,"minute":0}}
                          ],
            "test":[{"start":{"day":20,"hour":0,"minute":0},
                     "end":{"day":21,"hour":22,"minute":0}},
                    ]
        },
        # SPLIT 8
        {
            "train":[{"start":{"day":26,"hour":0,"minute":0},
                      "end":{"day":28,"hour":22,"minute":0}},
                     {"start":{"day":2,"hour":0,"minute":0},
                      "end":{"day":18,"hour":22,"minute":0}}
                     ],
            "validation":[{"start":{"day":20,"hour":0,"minute":0},
                           "end":{"day":21,"hour":22,"minute":0}}
                          ],
            "test":[{"start":{"day":23,"hour":0,"minute":0},
                     "end":{"day":24,"hour":22,"minute":0}},
                    ]
        },

    ]
    return INDEXS[num_split]
