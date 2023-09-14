import json
import os
import polars as pl
import array
import pathlib
def make_analysis(
        *,
        root_dir : str = "Resources/Resultados/",
        LowData : bool = True
):
    pl_resultados = pl.DataFrame()
    path = "Resources/Resultados/Agregado/LowData/Split0/minute/info_resultados.txt"
    with open(path,"r") as f:
        data = json.load(f)
    pl_parcial = pl.DataFrame([data],infer_schema_length=1)
    if pl_resultados.is_empty():
        pl_resultados = pl_parcial
    else:
        pl_resultados = pl_resultados.extend(pl_parcial)

    # Select not nested columns
    pl_resultados = pl_resultados.drop("HIST")
    not_nested_columns = [col for col in pl_resultados.columns if not pl_resultados[col].is_numeric()]
    # Create a new DataFrame with only the not nested columns
    path_analisis = 'Resources/Resultados/analisis.csv'
    pl_resultados = pl_resultados.drop(not_nested_columns)
    pl_resultados.write_csv(path_analisis,sep=",")
    return

if __name__ == "__main__":
    make_analysis()