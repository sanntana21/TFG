import json
import logging
import os
import polars as pl
import array
import pathlib
def make_analysis(
        *,
        root_dir : str = "Resources/Resultados/",
        LowData : bool = True,
        computed_option : int = None
):
    try:
        if computed_option > 2:
            raise ValueError("Computed option no disponible")

        str_comunted_option = "minute" if computed_option == 0 else "hour"
        extra_path = "/LowData" if LowData else ""
        pl_resultados = pl.DataFrame()
        for model in ["Agregado","Individual","Matrix"][1:2]:
            path_analisis = f"Resources/Resultados/{model}{extra_path}/analisis_{str_comunted_option}.csv"
            for split in range(0,9):
                path = f"Resources/Resultados/{model}{extra_path}/Split{split}/{str_comunted_option}/info_resultados.txt"
                with open(path,"r") as f:
                    data = json.load(f)

                data = {k:eval(v) for k,v in data.items()}

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

            data_per_column = { col:[pl_resultados[col].mean(),pl_resultados[col].std()] for col in pl_resultados.columns}
            pl_resultados.extend(pl.DataFrame(data_per_column,infer_schema_length=len(list(data_per_column.values())[0])))
            pl_resultados.insert_at_idx(0,pl.Series("split",["0","1","2","3","4","5","6","7","8","Mean","Std"]))
            pl_resultados.write_csv(path_analisis,sep=",")

    except BaseException as e:
        logging.error(f"Error durante la lectura del modelo {model} y el split{split}: {e}")
    return

if __name__ == "__main__":
    make_analysis(computed_option=0)