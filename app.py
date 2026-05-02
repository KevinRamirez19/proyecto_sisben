from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)

# ── Cargar y preparar datos ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

fact     = pd.read_csv(os.path.join(DATA, 'fact_persona.csv'),   encoding='utf-8-sig')
sisben   = pd.read_csv(os.path.join(DATA, 'dim_sisben.csv'),     encoding='utf-8-sig')
persona  = pd.read_csv(os.path.join(DATA, 'dim_persona.csv'),    encoding='utf-8-sig')
municipio= pd.read_csv(os.path.join(DATA, 'dim_municipio.csv'),  encoding='utf-8-sig')
zona     = pd.read_csv(os.path.join(DATA, 'dim_zona.csv'),       encoding='utf-8-sig')

df = (fact
      .merge(sisben,    on='sisben_sk')
      .merge(persona,   on='persona_sk')
      .merge(municipio, on='municipio_sk')
      .merge(zona,      on='zona_sk'))

CARENCIAS = ['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15']
CARENCIA_LABELS = {
    'I1': 'Sin logro educativo','I2': 'Analfabetismo','I3': 'Asistencia escolar',
    'I4': 'Rezago escolar','I5': 'Trabajo infantil','I6': 'Desempleo de larga duración',
    'I7': 'Empleo informal','I8': 'Sin aseguramiento en salud','I9': 'Sin acceso a servicios de salud',
    'I10': 'Sin cobertura en protección a la vejez','I11': 'Sin acceso a fuente de agua mejorada',
    'I12': 'Inadecuada eliminación de excretas','I13': 'Pisos inadecuados',
    'I14': 'Paredes exteriores inadecuadas','I15': 'Hacinamiento crítico'
}

# ── Rutas ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    stats = {
        'total':   f"{len(df):,}",
        'pobres':  f"{df['es_pobre_ipm'].sum():,}",
        'pct_pob': f"{df['es_pobre_ipm'].mean()*100:.1f}%",
        'ipm_avg': f"{df['ipm_score'].mean():.2f}",
        'municipios': str(df['nom_municipio'].nunique()),
        'clusters': str(df['cluster_kmeans'].nunique())
    }
    return render_template('index.html', stats=stats)


# ── API: Gráfica 1 – Distribución SISBEN ────────────────────────────────────
@app.route('/api/sisben_dist')
def api_sisben():
    counts = df['grupo_desc'].value_counts().reset_index()
    counts.columns = ['grupo','total']
    order = ['A – Pobreza extrema','B – Pobreza moderada','C – Vulnerable','D – No pobre']
    counts = counts.set_index('grupo').reindex(order).reset_index()
    return jsonify(counts.to_dict(orient='list'))


# ── API: Gráfica 2 – Carencias acumuladas ───────────────────────────────────
@app.route('/api/carencias')
def api_carencias():
    totals = df[CARENCIAS].sum().reset_index()
    totals.columns = ['indicador','total']
    totals['label'] = totals['indicador'].map(CARENCIA_LABELS)
    totals = totals.sort_values('total', ascending=False)
    return jsonify(totals.to_dict(orient='list'))


# ── API: Gráfica 3 – Clusters KMeans ────────────────────────────────────────
@app.route('/api/kmeans')
def api_kmeans():
    clust = df.groupby('cluster_kmeans')['grupo_desc'].value_counts().unstack(fill_value=0)
    result = {'clusters': clust.index.tolist()}
    for col in clust.columns:
        result[col] = clust[col].tolist()
    return jsonify(result)


# ── API: Gráfica 4 – IPM score por grupo SISBEN ──────────────────────────────
@app.route('/api/ipm_boxplot')
def api_ipm():
    order = ['A – Pobreza extrema','B – Pobreza moderada','C – Vulnerable','D – No pobre']
    result = {}
    for g in order:
        vals = df[df['grupo_desc'] == g]['ipm_score'].tolist()
        q1,med,q3 = np.percentile(vals,[25,50,75])
        iqr = q3 - q1
        result[g] = {
            'min': float(max(min(vals), q1 - 1.5*iqr)),
            'q1': float(q1), 'median': float(med),
            'q3': float(q3),
            'max': float(min(max(vals), q3 + 1.5*iqr)),
            'mean': float(np.mean(vals))
        }
    return jsonify(result)


# ── API: Gráfica 5 – Heatmap correlación carencias ──────────────────────────
@app.route('/api/correlacion')
def api_correlacion():
    corr = df[CARENCIAS].corr().round(3)
    return jsonify({
        'labels': [CARENCIA_LABELS[c] for c in CARENCIAS],
        'keys': CARENCIAS,
        'matrix': corr.values.tolist()
    })


# ── API: Gráfica 6 – Actividad económica vs pobreza ─────────────────────────
@app.route('/api/actividad_pobreza')
def api_actividad():
    cross = df.groupby(['actividad_economica','es_pobre_ipm']).size().unstack(fill_value=0)
    cross.columns = ['No pobre','Pobre']
    cross = cross.sort_values('Pobre', ascending=False).head(8)
    return jsonify({
        'actividades': cross.index.tolist(),
        'pobre': cross['Pobre'].tolist(),
        'no_pobre': cross['No pobre'].tolist()
    })


# ── API: Gráfica 7 – Carencias por cluster ──────────────────────────────────
@app.route('/api/radar_clusters')
def api_radar():
    result = {}
    for c in sorted(df['cluster_kmeans'].unique()):
        sub = df[df['cluster_kmeans'] == c]
        result[f'Cluster {c}'] = [round(sub[i].mean(), 3) for i in CARENCIAS]
    result['labels'] = [CARENCIA_LABELS[i] for i in CARENCIAS]
    return jsonify(result)


# ── API: Gráfica 8 – IPM score por municipio (top 15) ───────────────────────
@app.route('/api/ipm_municipio')
def api_ipm_muni():
    top = (df.groupby('nom_municipio')['ipm_score']
             .mean()
             .sort_values(ascending=False)
             .head(15)
             .reset_index())
    top.columns = ['municipio','ipm_promedio']
    return jsonify(top.to_dict(orient='list'))


# ── API: Gráfica 9 – Género vs grupo SISBEN ─────────────────────────────────
@app.route('/api/genero_sisben')
def api_genero():
    cross = df.groupby(['sexo','grupo_desc']).size().unstack(fill_value=0)
    order = ['A – Pobreza extrema','B – Pobreza moderada','C – Vulnerable','D – No pobre']
    cross = cross.reindex(columns=order, fill_value=0)
    return jsonify({
        'sexos': cross.index.tolist(),
        'grupos': cross.columns.tolist(),
        'values': cross.values.tolist()
    })


# ── API: Estadísticas KMeans ─────────────────────────────────────────────────
@app.route('/api/kmeans_stats')
def api_kmeans_stats():
    stats = df.groupby('cluster_kmeans').agg(
        total=('persona_fact_sk','count'),
        ipm_mean=('ipm_score','mean'),
        pct_pobre=('es_pobre_ipm','mean')
    ).reset_index()
    stats['ipm_mean'] = stats['ipm_mean'].round(2)
    stats['pct_pobre'] = (stats['pct_pobre']*100).round(1)
    return jsonify(stats.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
