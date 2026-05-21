import os
import sys
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, round, desc, when


os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

def get_spark_session():
    spark = SparkSession.builder \
        .appName("AnalisisSocioeconomico") \
        .master("local[1]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.ui.enabled", "false") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark



def cargar_datos(spark, ruta="./data/"):
    dim_zona       = spark.read.csv(ruta + "dim_zona.csv",       header=True, inferSchema=True)
    dim_municipio  = spark.read.csv(ruta + "dim_municipio.csv",  header=True, inferSchema=True)
    dim_persona    = spark.read.csv(ruta + "dim_persona.csv",    header=True, inferSchema=True)
    dim_sisben     = spark.read.csv(ruta + "dim_sisben.csv",     header=True, inferSchema=True)
    dim_tiempo     = spark.read.csv(ruta + "dim_tiempo.csv",     header=True, inferSchema=True)


    fact           = spark.read.csv(ruta + "fact_caracteristicas.csv",   header=True, inferSchema=True)

    df = fact \
        .join(dim_municipio, on="municipio_sk", how="left") \
        .join(dim_persona,   on="persona_sk",   how="left") \
        .join(dim_sisben,    on="sisben_sk",     how="left") \
        .join(dim_zona,      on="zona_sk",       how="left") \
        .join(dim_tiempo,    on="tiempo_sk",     how="left")

    return df



def obtener_resultados(df):
    pobreza_departamento = (
        df.groupBy("nom_departamento")
          .agg(
              count("persona_fact_sk").alias("total_personas"),
              sum("es_pobre_ipm").alias("pobres_ipm"),
              round(avg("ipm_score"), 2).alias("score_ipm_promedio")
          )
          .withColumn(
              "tasa_pobreza_pct",
              round(col("pobres_ipm") / col("total_personas") * 100, 2)
          )
          .orderBy(desc("tasa_pobreza_pct"))
          .toPandas()
          .to_dict(orient="records")
    )

    distribucion_sisben = (
        df.groupBy("grupo_desc", "clasificacion_desc")
          .agg(count("persona_fact_sk").alias("total_personas"))
          .orderBy("grupo_desc", "clasificacion_desc")
          .toPandas()
          .to_dict(orient="records")
    )

    pobreza_zona = (
        df.groupBy("zona_desc")
          .agg(
              count("persona_fact_sk").alias("total_personas"),
              sum("es_pobre_ipm").alias("pobres_ipm"),
              round(avg("ipm_score"), 2).alias("score_ipm_promedio")
          )
          .withColumn(
              "tasa_pobreza_pct",
              round(col("pobres_ipm") / col("total_personas") * 100, 2)
          )
          .orderBy(desc("tasa_pobreza_pct"))
          .toPandas()
          .to_dict(orient="records")
    )

    educacion_pobreza = (
        df.groupBy("nivel_educativo")
          .agg(
              count("persona_fact_sk").alias("total_personas"),
              round(avg("ipm_score"), 2).alias("score_ipm_promedio"),
              round(
                  sum("es_pobre_ipm") / count("persona_fact_sk") * 100, 2
              ).alias("tasa_pobreza_pct")
          )
          .orderBy(desc("tasa_pobreza_pct"))
          .toPandas()
          .to_dict(orient="records")
    )


    actividad_pension = (
        df.groupBy("actividad_economica", "cotiza_pension")
          .agg(count("persona_fact_sk").alias("total_personas"))
          .orderBy("actividad_economica", "cotiza_pension")
          .toPandas()
          .to_dict(orient="records")
    )

    municipios_mas_pobres = (
        df.groupBy("nom_municipio", "nom_departamento")
          .agg(
              count("persona_fact_sk").alias("total_personas"),
              round(avg("ipm_score"), 2).alias("score_ipm_promedio"),
              round(
                  sum("es_pobre_ipm") / count("persona_fact_sk") * 100, 2
              ).alias("tasa_pobreza_pct")
          )
          .orderBy(desc("score_ipm_promedio"))
          .limit(10)
          .toPandas()
          .to_dict(orient="records")
    )

    evolucion_temporal = (
        df.groupBy("anio")
          .agg(
              count("persona_fact_sk").alias("total_personas"),
              round(
                  sum("es_pobre_ipm") / count("persona_fact_sk") * 100, 2
              ).alias("tasa_pobreza_pct"),
              round(avg("ipm_score"), 2).alias("score_ipm_promedio")
          )
          .orderBy("anio")
          .toPandas()
          .to_dict(orient="records")
    )

    return {
        "pobreza_departamento":  pobreza_departamento,
        "distribucion_sisben":   distribucion_sisben,
        "pobreza_zona":          pobreza_zona,
        "educacion_pobreza":     educacion_pobreza,
        "actividad_pension":     actividad_pension,
        "municipios_mas_pobres": municipios_mas_pobres,
        "evolucion_temporal":    evolucion_temporal,
    }




if __name__ == "__main__":


    RUTA_CSV = "./data/"

    spark = get_spark_session()
    df    = cargar_datos(spark, ruta=RUTA_CSV)

   