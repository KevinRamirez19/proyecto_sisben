from flask import Flask, render_template, jsonify, send_from_directory
import importlib.util
import pandas as pd
import numpy as np
import json
import os
import folium
app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')


fact      = pd.read_csv(os.path.join(DATA, 'fact_caracteristicas.csv'),  encoding='utf-8-sig')
sisben    = pd.read_csv(os.path.join(DATA, 'dim_sisben.csv'),    encoding='utf-8-sig')
persona   = pd.read_csv(os.path.join(DATA, 'dim_persona.csv'),   encoding='utf-8-sig')
municipio = pd.read_csv(os.path.join(DATA, 'dim_municipio.csv'), encoding='utf-8-sig')
zona      = pd.read_csv(os.path.join(DATA, 'dim_zona.csv'),      encoding='utf-8-sig')

df = (fact
      .merge(sisben,    on='sisben_sk')
      .merge(persona,   on='persona_sk')
      .merge(municipio, on='municipio_sk')
      .merge(zona,      on='zona_sk'))

print(f" Datos cargados: {len(df):,} registros")

CARENCIAS = ['I1','I2','I3','I4','I5','I6','I7',
             'I8','I9','I10','I11','I12','I13','I14','I15']

CARENCIA_LABELS = {
    'I1':  'Sin logro educativo',
    'I2':  'Analfabetismo',
    'I3':  'Asistencia escolar',
    'I4':  'Rezago escolar',
    'I5':  'Trabajo infantil',
    'I6':  'Desempleo de larga duración',
    'I7':  'Empleo informal',
    'I8':  'Sin aseguramiento en salud',
    'I9':  'Sin acceso a servicios de salud',
    'I10': 'Sin cobertura en protección a la vejez',
    'I11': 'Sin acceso a fuente de agua mejorada',
    'I12': 'Inadecuada eliminación de excretas',
    'I13': 'Pisos inadecuados',
    'I14': 'Paredes exteriores inadecuadas',
    'I15': 'Hacinamiento crítico',
}

STATS = {
    'total':      f"{len(df):,}",
    'pobres':     f"{int(df['es_pobre_ipm'].sum()):,}",
    'pct_pob':    f"{df['es_pobre_ipm'].mean()*100:.1f}%",
    'ipm_avg':    f"{df['ipm_score'].mean():.2f}",
    'municipios': str(df['nom_municipio'].nunique()),
    'clusters':   str(df['cluster_kmeans'].nunique()),
    'no_pobres':  f"{int((df['es_pobre_ipm'] == 0).sum()):,}",
    'carencia_top': CARENCIA_LABELS[df[CARENCIAS].sum().idxmax()],
}


@app.route('/')
def index():
    return render_template('index.html', stats=STATS)

@app.route('/inicio')
def inicio():
    return index()

@app.route('/pgc')
def pgc():
    return render_template('pgc.html', stats=STATS)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', stats=STATS)

@app.route('/datos')
def datos():
    meta = {
        'registros':  f"{len(df):,}",
        'municipios': str(df['nom_municipio'].nunique()),
        'variables':  str(len(df.columns)),
        'carencias':  str(len(CARENCIAS)),
        'anio':       '2024 · S1',
        'fuente':     'DNP – datos.gov.co',
    }
    return render_template('datos.html', stats=STATS, meta=meta)

@app.route('/dashboard2')
def dashboard2():
    return render_template('dashboard2.html', stats=STATS, active='dashboard2')

@app.route('/kmeans')
def kmeans():
    return render_template('kmeans.html', stats=STATS)

@app.route('/dashboard/spark')
def dashboard_spark():
    return render_template('dashboard_pyspark.html', stats=STATS, active='spark')

@app.route('/api/spark')
def api_spark():

    # 1. Pobreza por departamento
    dept = df.groupby('nom_departamento').agg(
        total_personas=('persona_fact_sk', 'count'),
        pobres_ipm=('es_pobre_ipm', 'sum'),
        score_ipm_promedio=('ipm_score', 'mean')
    ).reset_index()
    dept['tasa_pobreza_pct']   = (dept['pobres_ipm'] / dept['total_personas'] * 100).round(2)
    dept['score_ipm_promedio'] = dept['score_ipm_promedio'].round(2)

    # 2. Distribución SISBÉN
    sisb = df.groupby(['grupo_desc', 'clasificacion_desc']).agg(
        total_personas=('persona_fact_sk', 'count')
    ).reset_index()

    # 3. Pobreza por zona
    zona_g = df.groupby('zona_desc').agg(
        total_personas=('persona_fact_sk', 'count'),
        pobres_ipm=('es_pobre_ipm', 'sum'),
        score_ipm_promedio=('ipm_score', 'mean')
    ).reset_index()
    zona_g['tasa_pobreza_pct']   = (zona_g['pobres_ipm'] / zona_g['total_personas'] * 100).round(2)
    zona_g['score_ipm_promedio'] = zona_g['score_ipm_promedio'].round(2)

    # 4. Educación vs pobreza
    edu = df.groupby('nivel_educativo').agg(
        total_personas=('persona_fact_sk', 'count'),
        score_ipm_promedio=('ipm_score', 'mean'),
        tasa_pobreza_pct=('es_pobre_ipm', 'mean')
    ).reset_index()
    edu['tasa_pobreza_pct']   = (edu['tasa_pobreza_pct'] * 100).round(2)
    edu['score_ipm_promedio'] = edu['score_ipm_promedio'].round(2)

    # 5. Actividad económica con tasa de pobreza
    act = df.groupby('actividad_economica').agg(
        total_personas=('persona_fact_sk', 'count'),
        pobres_ipm=('es_pobre_ipm', 'sum'),
        tasa_pobreza_pct=('es_pobre_ipm', 'mean')
    ).reset_index()
    act['tasa_pobreza_pct'] = (act['tasa_pobreza_pct'] * 100).round(2)

    # 6. Top 10 municipios más pobres
    muni = df.groupby(['nom_municipio', 'nom_departamento']).agg(
        total_personas=('persona_fact_sk', 'count'),
        score_ipm_promedio=('ipm_score', 'mean'),
        pobres_ipm=('es_pobre_ipm', 'sum')
    ).reset_index()
    muni['tasa_pobreza_pct']   = (muni['pobres_ipm'] / muni['total_personas'] * 100).round(2)
    muni['score_ipm_promedio'] = muni['score_ipm_promedio'].round(2)
    muni = muni[muni['nom_municipio'] != 'Pendiente DIVIPOLA']  # ← filtra filas sin municipio
    muni = muni.sort_values('score_ipm_promedio', ascending=False).head(10)

    # 7. Evolución temporal — merge limpio con solo anio
    try:
        tiempo = pd.read_csv(os.path.join(DATA, 'dim_tiempo.csv'), encoding='utf-8-sig')
        tiempo_clean = tiempo[['tiempo_sk', 'anio']].drop_duplicates('tiempo_sk')
        df_t = df.merge(tiempo_clean, on='tiempo_sk', how='left')
        evo = df_t.groupby('anio').agg(
            total_personas=('persona_fact_sk', 'count'),
            pobres_ipm=('es_pobre_ipm', 'sum'),
            score_ipm_promedio=('ipm_score', 'mean')
        ).reset_index()
        evo['tasa_pobreza_pct']   = (evo['pobres_ipm'] / evo['total_personas'] * 100).round(2)
        evo['score_ipm_promedio'] = evo['score_ipm_promedio'].round(2)
        evo = evo.to_dict(orient='records')
    except Exception as e:
        print("ERROR temporal:", e)
        evo = []

    return jsonify({
        'pobreza_departamento':  dept.to_dict(orient='records'),
        'distribucion_sisben':   sisb.to_dict(orient='records'),
        'pobreza_zona':          zona_g.to_dict(orient='records'),
        'educacion_pobreza':     edu.to_dict(orient='records'),
        'actividad_pension':     act.to_dict(orient='records'),
        'municipios_mas_pobres': muni.to_dict(orient='records'),
        'evolucion_temporal':    evo,
    })



def cargar_modulo_spark():
    ruta_modulo = os.path.join(BASE, 'spark.py')
    spec = importlib.util.spec_from_file_location('spark_module', ruta_modulo)
    spark_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spark_module)
    return spark_module


@app.route('/spark', endpoint='spark')
def spark_page():
    error = None
    resultados = None
    spark_session = None
    try:
        spark_module  = cargar_modulo_spark()
        spark_session = spark_module.get_spark_session()
        ruta_data     = os.path.join(BASE, 'data') + os.sep
        df_spark      = spark_module.cargar_datos(spark_session, ruta=ruta_data)
        resultados    = spark_module.obtener_resultados(df_spark)
    except Exception as exc:
        error = str(exc)
    finally:
        if spark_session is not None:
            try:
                spark_session.stop()
            except Exception:
                pass
    return render_template('spark.html', resultados=resultados, error=error, active='spark')



@app.route('/api/sisben_dist')
def api_sisben():
    counts = df['grupo_desc'].value_counts().reset_index()
    counts.columns = ['grupo', 'total']
    order = ['A – Pobreza extrema', 'B – Pobreza moderada',
             'C – Vulnerable', 'D – No pobre']
    counts = counts.set_index('grupo').reindex(order).reset_index()
    return jsonify(counts.to_dict(orient='list'))


@app.route('/api/carencias')
def api_carencias():
    totals = df[CARENCIAS].sum().reset_index()
    totals.columns = ['indicador', 'total']
    totals['label'] = totals['indicador'].map(CARENCIA_LABELS)
    totals = totals.sort_values('total', ascending=False)
    return jsonify(totals.to_dict(orient='list'))


@app.route('/api/kmeans')
def api_kmeans():
    clust = (df.groupby('cluster_kmeans')['grupo_desc']
            .value_counts()
            .unstack(fill_value=0))
    result = {'clusters': clust.index.tolist()}
    for col in clust.columns:
        result[col] = clust[col].tolist()
    return jsonify(result)


@app.route('/api/ipm_boxplot')
def api_ipm():
    order = ['A – Pobreza extrema', 'B – Pobreza moderada',
            'C – Vulnerable', 'D – No pobre']
    result = {}
    for g in order:
        vals = df[df['grupo_desc'] == g]['ipm_score'].dropna().tolist()
        if not vals:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        result[g] = {
            'min':    float(max(min(vals), q1 - 1.5 * iqr)),
            'q1':     float(q1),
            'median': float(med),
            'q3':     float(q3),
            'max':    float(min(max(vals), q3 + 1.5 * iqr)),
            'mean':   float(np.mean(vals)),
        }
    return jsonify(result)


@app.route('/api/correlacion')
def api_correlacion():
    corr = df[CARENCIAS].corr().round(3)
    return jsonify({
        'labels': [CARENCIA_LABELS[c] for c in CARENCIAS],
        'keys':   CARENCIAS,
        'matrix': corr.values.tolist(),
    })


@app.route('/api/actividad_pobreza')
def api_actividad():
    cross = (df.groupby(['actividad_economica', 'es_pobre_ipm'])
               .size()
               .unstack(fill_value=0))
    cross.columns = ['No pobre', 'Pobre']
    cross = cross.sort_values('Pobre', ascending=False).head(8)
    return jsonify({
        'actividades': cross.index.tolist(),
        'pobre':       cross['Pobre'].tolist(),
        'no_pobre':    cross['No pobre'].tolist(),
    })


@app.route('/api/radar_clusters')
def api_radar():
    result = {}
    for c in sorted(df['cluster_kmeans'].unique()):
        sub = df[df['cluster_kmeans'] == c]
        result[f'Cluster {c}'] = [round(float(sub[i].mean()), 3) for i in CARENCIAS]
    result['labels'] = [CARENCIA_LABELS[i] for i in CARENCIAS]
    return jsonify(result)


@app.route('/api/ipm_municipio')
def api_ipm_muni():
    top = (df.groupby('nom_municipio')['ipm_score']
             .mean()
             .sort_values(ascending=False)
             .head(15)
             .reset_index())
    top.columns = ['municipio', 'ipm_promedio']
    top['ipm_promedio'] = top['ipm_promedio'].round(2)
    return jsonify(top.to_dict(orient='list'))


@app.route('/api/genero_sisben')
def api_genero():
    cross = df.groupby(['sexo', 'grupo_desc']).size().unstack(fill_value=0)
    order = ['A – Pobreza extrema', 'B – Pobreza moderada',
             'C – Vulnerable', 'D – No pobre']
    cross = cross.reindex(columns=order, fill_value=0)
    return jsonify({
        'sexos':  cross.index.tolist(),
        'grupos': cross.columns.tolist(),
        'values': cross.values.tolist(),
    })


@app.route('/api/kmeans_stats')
def api_kmeans_stats():
    stats = df.groupby('cluster_kmeans').agg(
        total=('persona_fact_sk', 'count'),
        ipm_mean=('ipm_score', 'mean'),
        pct_pobre=('es_pobre_ipm', 'mean'),
    ).reset_index()
    stats['ipm_mean']  = stats['ipm_mean'].round(2)
    stats['pct_pobre'] = (stats['pct_pobre'] * 100).round(1)
    return jsonify(stats.to_dict(orient='records'))


@app.route('/api/zona_sisben')
def api_zona_sisben():
    cross = df.groupby(['zona_desc', 'grupo_desc']).size().unstack(fill_value=0)
    order = ['A – Pobreza extrema', 'B – Pobreza moderada',
             'C – Vulnerable', 'D – No pobre']
    cross = cross.reindex(columns=order, fill_value=0)
    return jsonify({
        'zonas':  cross.index.tolist(),
        'grupos': cross.columns.tolist(),
        'values': cross.values.tolist(),
    })


@app.route('/api/alfabetismo')
def api_alfabetismo():
    counts = df['alfabetismo'].value_counts().reset_index()
    counts.columns = ['tipo', 'total']
    return jsonify(counts.to_dict(orient='list'))


@app.route('/api/stats')
def api_stats():
    return jsonify(STATS)

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html', stats=STATS), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Error interno del servidor', 'detalle': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')