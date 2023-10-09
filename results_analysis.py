import json
import logging
import os
import polars as pl
import array
import pathlib


def make_analysis(
        *,
        root_dir : str = "Resources/Resultados/",
        # num_model : int = 0,
        LowData : bool = True,
        computed_option : int = None
):
    try:
        computed_option_str = ""
        if computed_option == 1:
            computed_option_str = "hour"
        else:
            computed_option_str = "minute"
        # model = ["Agregado","Individual","Matrix"][num_model]
        if computed_option > 2:
            raise ValueError("Computed option no disponible")

        str_comunted_option = "minute" if computed_option == 0 else "hour"
        extra_path = "/LowData" if LowData else ""
        pl_resultados = pl.DataFrame()
        # path_analisis = f"Resources/Resultados/{model}{extra_path}/analisis_{str_comunted_option}.csv"
        eje = pl.Series("split",["0","1","2","3","4","5","6","7","8","Mean","Std"])
        pl_point_to_point = pl.DataFrame(eje)
        pl_tarjet = pl.DataFrame(eje)
        pl_best = pl.DataFrame(eje)
        pl_worst = pl.DataFrame(eje)
        pl_average = pl.DataFrame(eje)
        pl_two_days = pl.DataFrame(eje)
        pl_prueba = pl.DataFrame(eje)

        for model in ["Agregado","Individual","Matrix"]:
            pl_resultados = pl.DataFrame()
            for split in range(0,9):
                path = f"Resources/Resultados/{model}{extra_path}/Split{split}/{str_comunted_option}/info_resultados.txt"
                with open(path,"r") as f:
                    data = json.load(f)

                data = {k:eval(v) if isinstance(v,str) else v for k,v in data.items()}

                pl_parcial = pl.DataFrame([data],infer_schema_length=1)
                pl_parcial = pl_parcial.with_columns(
                    [pl.col("BEST_OLS").arr.explode().take(0).cast(pl.Float64).alias("BEST_OLS_B"),
                    pl.col("BEST_OLS").arr.explode().take(1).cast(pl.Float64).alias("BEST_OLS_M"),
                    pl.col("WORST_OLS").arr.explode().take(0).cast(pl.Float64).alias("WORST_OLS_B"),
                    pl.col("WORST_OLS").arr.explode().take(1).cast(pl.Float64).alias("WORST_OLS_M"),
                    pl.col("MEDIUM_OLS").arr.explode().take(0).cast(pl.Float64).alias("MEDIUM_OLS_B"),
                    pl.col("MEDIUM_OLS").arr.explode().take(1).cast(pl.Float64).alias("MEDIUM_OLS_M"),
                    pl.col("TWO_DAYS_OLS").arr.explode().take(0).cast(pl.Float64).alias("TWO_DAYS_OLS_B"),
                    pl.col("TWO_DAYS_OLS").arr.explode().take(1).cast(pl.Float64).alias("TWO_DAYS_OLS_M"),
                    pl.col("TRANSFORMED_PREDICITION_OLS").arr.explode().take(0).cast(pl.Float64).alias("TRANSFORMED_PREDICTION_B"),
                    pl.col("TRANSFORMED_PREDICITION_OLS").arr.explode().take(1).cast(pl.Float64).alias("TRANSFORMED_PREDICTION_M")
                    ]
                )
                #Medium OLS , WORTS
                if pl_resultados.is_empty():
                    pl_resultados = pl_parcial
                else:
                    pl_resultados = pl_resultados.extend(pl_parcial)

            pl_resultados = pl_resultados.drop(
                ["TRANSFORMED_PREDICITION_OLS","HIST","BEST_OLS","MEDIUM_OLS","WORST_OLS","TWO_DAYS_OLS"])
            # Create a new DataFrame with only the not nested columns
            pl_resultados =pl_resultados.rename({"POBLATIONAL_MAE":"TARGET_MAE","POBLATIONAL_MSE":"TARGET_MSE"})

            data_per_column = { col:[pl_resultados[col].mean(),pl_resultados[col].std()] for col in pl_resultados.columns}
            pl_resultados.extend(pl.DataFrame(data_per_column,infer_schema_length=len(list(data_per_column.values())[0])))
            pl_resultados.insert_at_idx(0,pl.Series("split",["0","1","2","3","4","5","6","7","8","Mean","Std"]))
            path_analisis = f"Resources/Resultados/{model}{extra_path}/analisis_{str_comunted_option}.csv"
            pl_resultados.write_csv(path_analisis,sep=",")
            model = "Matricial" if model == "Matrix" else model
            pl_point_to_point = pl_point_to_point.with_columns(pl_resultados["POINT_TO_POINT_MAE"].rename(f"MAE {model}"))
            pl_point_to_point = pl_point_to_point.with_columns(pl_resultados["POINT_TO_POINT_MSE"].rename(f"MSE {model}"))
            pl_tarjet = pl_tarjet.with_columns(pl_resultados["TARGET_MAE"].rename(f"MAE {model}"))
            pl_tarjet = pl_tarjet.with_columns(pl_resultados["TARGET_MSE"].rename(f"MSE {model}"))

            pl_best = pl_best.with_columns(pl_resultados["BEST"].rename(f"{model}"))
            pl_worst = pl_worst.with_columns(pl_resultados["WORST"].rename(f"{model}"))
            pl_average = pl_average.with_columns(pl_resultados["MEDIUM"].rename(f"{model}"))
            pl_two_days = pl_two_days.with_columns(pl_resultados["TWO_DAYS_RESULTS_PREDICTED"].rename(f"{model}"))
            pl_prueba = pl_prueba.with_columns(pl_resultados["POBLATIONAL_MEAN"].rename(f"{model}"))

        path_analisis = f"Resources/Resultados/{computed_option_str}_analisis_POINT_TO_POINT.csv"
        pl_point_to_point.write_csv(path_analisis,sep=",")

        path_analisis = f"Resources/Resultados/{computed_option_str}_analisis_TARGET.csv"
        pl_tarjet.write_csv(path_analisis,sep=",")

        path_analisis = f"Resources/Resultados/{computed_option_str}_analisis_BEST.csv"
        pl_best.write_csv(path_analisis,sep=",")

        path_analisis = f"Resources/Resultados/{computed_option_str}_analisis_WORST.csv"
        pl_worst.write_csv(path_analisis,sep=",")

        path_analisis = f"Resources/Resultados/{computed_option_str}_analisis_MEDIUM.csv"
        pl_average.write_csv(path_analisis,sep=",")

        path_analisis = f"Resources/Resultados/{computed_option_str}_analisis_TWO_DAYS_RESULTS_PREDICTED.csv"
        pl_two_days.write_csv(path_analisis,sep=",")

    except BaseException as e:
        logging.error(f"Error durante la lectura del modelo {model} y el split{split}: {e}")
    return True

if __name__ == "__main__":
    for computed_option in [0,1]:
        make_analysis(computed_option=computed_option)