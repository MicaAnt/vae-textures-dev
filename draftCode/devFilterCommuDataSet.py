import csv
import os
import pandas as pd
import ast

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

csv_path = "../midiDataTest/commu_meta.csv"

metaDataCOMMU = pd.read_csv(csv_path)

#print(csv_path)

def filtro_chords_por_beat(df):
    return df[
        df.apply(lambda row: len(ast.literal_eval(row["chord_progressions"])[0]) /
                            int(row["time_signature"].split("/")[0]) /
                            row["num_measures"] == 2,
                 axis=1)
    ]
    print("❌ Linhas excluídas:")
    print(df[~mask])
    return df[mask]

print(filtro_chords_por_beat(metaDataCOMMU))

#print(metaDataCOMMU.head())
