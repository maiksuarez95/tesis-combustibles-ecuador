import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ============================
# CONFIGURACIÓN DE LA PÁGINA
# ============================
st.set_page_config(
    page_title="Predicción de Combustibles Ecuador",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# ESTILOS (TEMA OSCURO ESTABLE)
# ============================
st.markdown("""
<style>
.stApp {
    background-color: #0d0f12 !important;
    color: #ffffff !important;
}

h1, h2, h3, h4, h5, p, span, label, div {
    color: #e6e6e6;
}

section[data-testid="stSidebar"] {
    background-color: #111418 !important;
}

.block-container {
    background-color: #0d0f12 !important;
    padding-top: 2rem;
}

.stMetric {
    background-color: #1b1f24 !important;
    border-radius: 12px;
    padding: 14px !important;
    border: 1px solid #2a2f36;
}

button {
    background-color: #1f6feb !important;
    color: white !important;
    border-radius: 8px !important;
}

button:hover {
    background-color: #388bfd !important;
}

input, textarea {
    background-color: #161b22 !important;
    color: #e6e6e6 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================
# CABECERA
# ============================
st.markdown("""
# ⛽ Sistema de Predicción del Precio de Combustibles  
### Proyecto de Titulación – Michael Suárez  
---
""")

# ============================
# SIDEBAR
# ============================
st.sidebar.header("⚙️ Configuración")

archivo = st.sidebar.file_uploader(
    "📤 Cargar archivo CSV",
    type=["csv"],
    help="Debe contener columnas: ds,y"
)

horizonte_meses = st.sidebar.slider(
    "Horizonte de predicción (meses)",
    min_value=1,
    max_value=24,
    value=6,
    step=1
)

st.sidebar.markdown("---")

# ============================
# CARGA DE DATOS
# ============================
if archivo is not None:
    df = pd.read_csv(archivo)
    st.success("✅ Archivo cargado correctamente.")
else:
    df = pd.read_csv("data/gasolina_extra_ecuador.csv")
    st.info("ℹ️ Usando dataset real por defecto: Gasolina Extra automotriz en Ecuador.")

# Normalización y limpieza
df.columns = [c.strip().lower() for c in df.columns]

if "ds" not in df.columns or "y" not in df.columns:
    st.error("El archivo debe contener columnas llamadas 'ds' y 'y'.")
    st.stop()

df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna().sort_values("ds")

if df.empty:
    st.error("El dataset está vacío después de la limpieza.")
    st.stop()

# ============================
# CARGA Y PREPARACIÓN DE BRENT
# ============================
df_brent = pd.read_csv("data/brent_prices.csv")

df_brent.columns = [c.strip().lower() for c in df_brent.columns]

if "ds" not in df_brent.columns or "y" not in df_brent.columns:
    st.error("El archivo brent_prices.csv debe contener columnas llamadas 'ds' y 'y'.")
    st.stop()

df_brent["ds"] = pd.to_datetime(df_brent["ds"], errors="coerce")
df_brent["y"] = pd.to_numeric(df_brent["y"], errors="coerce")
df_brent = df_brent.dropna().sort_values("ds")

# Renombrar para evitar conflicto con la gasolina
df_brent = df_brent.rename(columns={"y": "brent"})

# Convertir Brent a frecuencia mensual (inicio de mes)
df_brent = (
    df_brent.set_index("ds")
    .resample("MS")
    .mean()
    .reset_index()
)

# Unir gasolina y Brent por fecha
df_comparacion = pd.merge(df, df_brent, on="ds", how="inner")

# ============================
# ESTADÍSTICAS HISTÓRICAS
# ============================
st.subheader("📊 Resumen histórico – Gasolina Extra (USD por galón)")

precio_medio = df["y"].mean()
precio_min = df["y"].min()
precio_max = df["y"].max()
precio_ultimo = df["y"].iloc[-1]
fecha_ultimo = df["ds"].iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Promedio", f"{precio_medio:.2f} USD/gal")
col2.metric("Mínimo", f"{precio_min:.2f} USD/gal")
col3.metric("Máximo", f"{precio_max:.2f} USD/gal")
col4.metric("Último valor", f"{precio_ultimo:.2f} USD/gal", fecha_ultimo.strftime("%Y-%m-%d"))

df_mostrar = df.copy()

df_mostrar = df_mostrar.rename(columns={
    "ds": "Fecha",
    "y": "Precio (USD/gal)"
})

df_mostrar["Fecha"] = pd.to_datetime(df_mostrar["Fecha"]).dt.strftime("%Y-%m-%d")

st.dataframe(df_mostrar, use_container_width=True)

# ============================
# MODELO PROPHET
# ============================
modelo = Prophet()
modelo.fit(df)

futuro = modelo.make_future_dataframe(periods=horizonte_meses, freq="MS")
pronostico = modelo.predict(futuro)

# Separar histórico y futuro
fecha_max_hist = df["ds"].max()
pred_hist = pronostico[pronostico["ds"] <= fecha_max_hist].copy()
pred_fut = pronostico[pronostico["ds"] > fecha_max_hist].copy()

# ============================
# EVALUACIÓN DEL MODELO
# ============================
st.subheader("📊 Evaluación del modelo (sobre datos históricos)")

df_eval = pred_hist.merge(df[["ds", "y"]], on="ds", how="inner")

y_real = df_eval["y"].values
y_pred = df_eval["yhat"].values

errores = y_real - y_pred
mae = np.mean(np.abs(errores))
mse = np.mean(errores ** 2)
rmse = np.sqrt(mse)

sst = np.sum((y_real - np.mean(y_real)) ** 2)
ssr = np.sum((y_real - y_pred) ** 2)
r2 = 1 - ssr / sst if sst != 0 else np.nan

colm1, colm2, colm3, colm4 = st.columns(4)
colm1.metric("MAE (Error Absoluto Medio)", f"{mae:.4f}")
colm2.metric("MSE (Error Cuadrático Medio)", f"{mse:.4f}")
colm3.metric("RMSE (Raíz del Error Cuadrático)", f"{rmse:.4f}")
colm4.metric("R² (Coeficiente de determinación)", f"{r2:.4f}")

# ============================
# MÉTRICAS FUTURAS
# ============================
if not pred_fut.empty:
    ultima_pred = pred_fut.tail(1)["yhat"].iloc[0]
    fecha_ult = pred_fut.tail(1)["ds"].iloc[0].strftime("%Y-%m-%d")

    colp1, colp2 = st.columns(2)
    colp1.metric("Predicción final", f"{ultima_pred:.2f} USD/galón", fecha_ult)
    colp2.metric("Variación vs último precio", f"{ultima_pred - precio_ultimo:.2f} USD/galón")

# ============================
# GRÁFICOS
# ============================
st.subheader("📈 Predicción del precio")

fig1 = plot_plotly(modelo, pronostico)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🔎 Tendencia y comportamiento estacional del precio")

fig2 = modelo.plot_components(pronostico)
fig2.set_size_inches(10, 6)
fig2.subplots_adjust(hspace=0.5)

# Fondo oscuro y textos claros
fig2.patch.set_facecolor("#0d0f12")

for ax in fig2.axes:
    ax.set_facecolor("#0d0f12")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

# Cambiar títulos y nombres de ejes
fig2.axes[0].set_title("Tendencia del precio", color="white")
fig2.axes[0].set_xlabel("Fecha")
fig2.axes[0].set_ylabel("Precio estimado (USD/galón)")

fig2.axes[1].set_title("Patrón estacional anual", color="white")
fig2.axes[1].set_xlabel("Mes del año")
fig2.axes[1].set_ylabel("Variación estacional")

st.pyplot(fig2)

st.markdown("""
📌 **Interpretación del gráfico**

- La parte superior muestra la **tendencia general del precio** a lo largo del tiempo.
- La parte inferior muestra el **comportamiento estacional**, es decir, cómo varía el precio según el momento del año.

**Cómo leerlo:**
- Valores positivos → meses en los que el precio tiende a ser más alto.
- Valores negativos → meses en los que el precio tiende a ser más bajo.

⚠️ Esta variación es relativa y forma parte del modelo; no representa precios exactos.
""")

# ============================
# COMPARACIÓN BRENT VS GASOLINA
# ============================
st.subheader("🌍 Relación entre Brent y Gasolina Extra")

if df_comparacion.empty:
    st.warning("No se encontraron fechas coincidentes entre Gasolina Extra y Brent.")
else:
    df_comp_mostrar = df_comparacion.copy()
    df_comp_mostrar["ds"] = df_comp_mostrar["ds"].dt.strftime("%Y-%m-%d")



fig_rel = go.Figure()

fig_rel.add_trace(
    go.Scatter(
        x=df_comparacion["ds"],
        y=df_comparacion["y"],
        name="Gasolina Extra (USD/galón)",
        mode="lines"
    )
)

fig_rel.add_trace(
    go.Scatter(
        x=df_comparacion["ds"],
        y=df_comparacion["brent"],
        name="Brent (USD/barril)",
        mode="lines",
        yaxis="y2"
    )
)

fig_rel.update_layout(
    title="Comparación de tendencias: Gasolina Extra vs Brent",
    xaxis=dict(title="Fecha"),
    yaxis=dict(title="Gasolina Extra (USD/galón)"),
    yaxis2=dict(
        title="Brent (USD/barril)",
        overlaying="y",
        side="right"
    ),
    template="plotly_dark",
    legend=dict(title="Serie")
)

st.plotly_chart(fig_rel, use_container_width=True)

correlacion = df_comparacion["y"].corr(df_comparacion["brent"])
st.metric("Correlación Brent vs Gasolina Extra", f"{correlacion:.2f}")

# Interpretación automática con colores
if correlacion < 0.3:
    st.error("🔴 Relación débil: poca dependencia entre el precio del Brent y la gasolina.")
elif correlacion < 0.7:
    st.warning("🟡 Relación moderada: existe cierta dependencia entre el precio del Brent y la gasolina.")
else:
    st.success("🟢 Relación fuerte: el precio del Brent influye significativamente en la gasolina.")


st.caption(
        "Nota: La gasolina Extra se expresa en USD/galón y el Brent en USD/barril. "
        "La comparación se presenta con fines de análisis de tendencia."
    )

# ============================
# DESCARGAS
# ============================
st.subheader("⬇️ Descarga de datos")

csv_hist = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Descargar históricos",
    csv_hist,
    "historicos.csv",
    "text/csv"
)

csv_pred = pred_fut[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Descargar predicción futura",
    csv_pred,
    "prediccion.csv",
    "text/csv"
)

# ============================
# MANUAL
# ============================
with st.expander("📘 Manual de uso"):
    st.markdown("""
1. Cargue un archivo CSV o use el dataset por defecto.  
2. Ajuste el horizonte de predicción.  
3. Revise las métricas de evaluación del modelo (MAE, MSE, RMSE, R²).  
4. Analice las gráficas de predicción.  
5. Descargue los datos para incluirlos en la tesis.
""")

# ============================
# FOOTER
# ============================
st.markdown("""
---
📌 *Desarrollado por **Michael Suárez** — Proyecto de Titulación Ecuador, 2026.*  
""")