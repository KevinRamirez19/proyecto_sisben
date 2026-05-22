import os
import sys
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    sum,
    avg,
    count,
    round as spark_round,
    desc,
    when
)

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)

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

    dim_zona = spark.read.csv(
        ruta + "dim_zona.csv",
        header=True,
        inferSchema=True
    )

    dim_municipio = spark.read.csv(
        ruta + "dim_municipio.csv",
        header=True,
        inferSchema=True
    )

    dim_persona = spark.read.csv(
        ruta + "dim_persona.csv",
        header=True,
        inferSchema=True
    )

    dim_sisben = spark.read.csv(
        ruta + "dim_sisben.csv",
        header=True,
        inferSchema=True
    )

    dim_tiempo = spark.read.csv(
        ruta + "dim_tiempo.csv",
        header=True,
        inferSchema=True
    )

    fact = spark.read.csv(
        ruta + "fact_caracteristicas.csv",
        header=True,
        inferSchema=True
    )

    df = fact \
        .join(dim_municipio, on="municipio_sk", how="left") \
        .join(dim_persona, on="persona_sk", how="left") \
        .join(dim_sisben, on="sisben_sk", how="left") \
        .join(dim_zona, on="zona_sk", how="left") \
        .join(dim_tiempo, on="tiempo_sk", how="left")

    return df

def obtener_resultados(df):

    pobreza_departamento = (
        df.groupBy("nom_departamento")
        .agg(
            count("persona_fact_sk").alias("total_personas"),
            sum("es_pobre_ipm").alias("pobres_ipm"),
            spark_round(avg("ipm_score"), 2).alias("score_ipm_promedio")
        )
        .withColumn(
            "tasa_pobreza_pct",
            spark_round(
                col("pobres_ipm") / col("total_personas") * 100,
                2
            )
        )
        .orderBy(desc("tasa_pobreza_pct"))
        .toPandas()
        .to_dict(orient="records")
    )

    distribucion_sisben = (
        df.groupBy("grupo_desc", "clasificacion_desc")
        .agg(
            count("persona_fact_sk").alias("total_personas")
        )
        .orderBy("grupo_desc", "clasificacion_desc")
        .toPandas()
        .to_dict(orient="records")
    )

    pobreza_zona = (
        df.groupBy("zona_desc")
        .agg(
            count("persona_fact_sk").alias("total_personas"),
            sum("es_pobre_ipm").alias("pobres_ipm"),
            spark_round(avg("ipm_score"), 2).alias("score_ipm_promedio")
        )
        .withColumn(
            "tasa_pobreza_pct",
            spark_round(
                col("pobres_ipm") / col("total_personas") * 100,
                2
            )
        )
        .orderBy(desc("tasa_pobreza_pct"))
        .toPandas()
        .to_dict(orient="records")
    )

    educacion_pobreza = (
        df.groupBy("nivel_educativo")
        .agg(
            count("persona_fact_sk").alias("total_personas"),
            spark_round(avg("ipm_score"), 2).alias("score_ipm_promedio"),
            spark_round(
                sum("es_pobre_ipm") / count("persona_fact_sk") * 100,
                2
            ).alias("tasa_pobreza_pct")
        )
        .orderBy(desc("tasa_pobreza_pct"))
        .toPandas()
        .to_dict(orient="records")
    )

    actividad_pension = (
        df.groupBy("actividad_economica", "cotiza_pension")
        .agg(
            count("persona_fact_sk").alias("total_personas")
        )
        .orderBy("actividad_economica", "cotiza_pension")
        .toPandas()
        .to_dict(orient="records")
    )

    municipios_mas_pobres = (
        df.groupBy("nom_municipio", "nom_departamento")
        .agg(
            count("persona_fact_sk").alias("total_personas"),
            spark_round(avg("ipm_score"), 2).alias("score_ipm_promedio"),
            spark_round(
                sum("es_pobre_ipm") / count("persona_fact_sk") * 100,
                2
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
            sum("es_pobre_ipm").alias("pobres_ipm"),
            spark_round(
                sum("es_pobre_ipm") / count("persona_fact_sk") * 100,
                2
            ).alias("tasa_pobreza_pct"),
            spark_round(avg("ipm_score"), 2).alias("score_ipm_promedio")
        )
        .orderBy("anio")
        .toPandas()
        .to_dict(orient="records")
    )

    return {
        "pobreza_departamento": pobreza_departamento,
        "distribucion_sisben": distribucion_sisben,
        "pobreza_zona": pobreza_zona,
        "educacion_pobreza": educacion_pobreza,
        "actividad_pension": actividad_pension,
        "municipios_mas_pobres": municipios_mas_pobres,
        "evolucion_temporal": evolucion_temporal
    }

def preparar_ml(df):

    indicadores = [
        "I1", "I2", "I3", "I4", "I5",
        "I6", "I7", "I8", "I9", "I10",
        "I11", "I12", "I13", "I14", "I15"
    ]

    assembler = VectorAssembler(
        inputCols=indicadores,
        outputCol="features_raw"
    )

    df = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,
        withStd=True
    )

    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    return df, indicadores

def ejecutar_kmeans(df):

    print("Ejecutando KMeans...")

    modelo = KMeans(
        k=4,
        seed=42,
        featuresCol="features",
        predictionCol="cluster"
    )

    inicio = time.time()

    fitted = modelo.fit(df)

    tiempo = round(time.time() - inicio, 2)

    pred = fitted.transform(df)

    perfil = (
        pred.groupBy("cluster")
        .agg(
            count("*").alias("n"),
            spark_round(avg("ipm_score"), 2).alias("ipm_promedio"),
            spark_round(avg("es_pobre_ipm") * 100, 2).alias("pct_pobre"),

            spark_round(avg("I1"), 3).alias("avg_I1"),
            spark_round(avg("I2"), 3).alias("avg_I2"),
            spark_round(avg("I3"), 3).alias("avg_I3"),
            spark_round(avg("I4"), 3).alias("avg_I4"),
            spark_round(avg("I5"), 3).alias("avg_I5")
        )
        .orderBy("cluster")
        .toPandas()
        .to_dict(orient="records")
    )

    dist_sisben = (
        pred.groupBy("cluster", "grupo_desc")
        .agg(count("*").alias("n"))
        .orderBy("cluster")
        .toPandas()
        .to_dict(orient="records")
    )

    return {
        "k": 4,
        "wssse": round(fitted.summary.trainingCost, 2),
        "tiempo_fit": tiempo,
        "perfil": perfil,
        "dist_sisben": dist_sisben
    }

def ejecutar_random_forest(df, indicadores):

    print("Ejecutando Random Forest...")

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    rf = RandomForestClassifier(
        labelCol="es_pobre_ipm",
        featuresCol="features",
        numTrees=50,
        maxDepth=8
    )

    inicio = time.time()

    model = rf.fit(train)

    tiempo = round(time.time() - inicio, 2)

    pred = model.transform(test)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="es_pobre_ipm",
        predictionCol="prediction"
    )

    accuracy = round(
        evaluator.evaluate(pred, {evaluator.metricName: "accuracy"}) * 100,
        2
    )

    f1 = round(
        evaluator.evaluate(pred, {evaluator.metricName: "f1"}) * 100,
        2
    )

    precision = round(
        evaluator.evaluate(pred, {evaluator.metricName: "weightedPrecision"}) * 100,
        2
    )

    recall = round(
        evaluator.evaluate(pred, {evaluator.metricName: "weightedRecall"}) * 100,
        2
    )

    importancias = []

    for idx, val in enumerate(model.featureImportances):
        importancias.append({
            "indicador": indicadores[idx],
            "valor": round(val * 100, 2)
        })

    importancias = sorted(
        importancias,
        key=lambda x: x["valor"],
        reverse=True
    )

    confusion = (
        pred.groupBy("es_pobre_ipm", "prediction")
        .agg(count("*").alias("count"))
        .toPandas()
        .to_dict(orient="records")
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tiempo_fit": tiempo,
        "importancias": importancias,
        "confusion": confusion
    }

def ejecutar_regresion(df, indicadores):

    print("Ejecutando Regresión Lineal...")

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    lr = LinearRegression(
        labelCol="ipm_score",
        featuresCol="features",
        maxIter=20,
        regParam=0.1
    )

    inicio = time.time()

    model = lr.fit(train)

    tiempo = round(time.time() - inicio, 2)

    pred = model.transform(test)

    evaluator_rmse = RegressionEvaluator(
        labelCol="ipm_score",
        predictionCol="prediction",
        metricName="rmse"
    )

    evaluator_r2 = RegressionEvaluator(
        labelCol="ipm_score",
        predictionCol="prediction",
        metricName="r2"
    )

    evaluator_mae = RegressionEvaluator(
        labelCol="ipm_score",
        predictionCol="prediction",
        metricName="mae"
    )

    rmse = round(evaluator_rmse.evaluate(pred), 4)
    r2 = round(evaluator_r2.evaluate(pred), 4)
    mae = round(evaluator_mae.evaluate(pred), 4)

    coeficientes = []

    for idx, coef in enumerate(model.coefficients):
        coeficientes.append({
            "indicador": indicadores[idx],
            "coef": round(float(coef), 4)
        })

    coeficientes = sorted(
        coeficientes,
        key=lambda x: abs(x["coef"]),
        reverse=True
    )

    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "intercepto": round(model.intercept, 4),
        "tiempo_fit": tiempo,
        "coeficientes": coeficientes
    }

if __name__ == "__main__":

    print("Cargando datos...")

    spark = get_spark_session()

    df = cargar_datos(spark)

    print("Generando análisis descriptivo...")

    resultados = obtener_resultados(df)

    print("Preparando Machine Learning...")

    df_ml, indicadores = preparar_ml(df)

    kmeans = ejecutar_kmeans(df_ml)

    random_forest = ejecutar_random_forest(df_ml, indicadores)

    regresion = ejecutar_regresion(df_ml, indicadores)

    resultados_ml = {
        "kmeans": kmeans,
        "random_forest": random_forest,
        "regresion_lineal": regresion
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=4)

    with open("ml_resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados_ml, f, ensure_ascii=False, indent=4)

    print("Archivos generados correctamente:")
    print(" resultados.json")
    print(" ml_resultados.json")

    spark.stop()