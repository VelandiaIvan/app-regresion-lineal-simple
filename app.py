"""Asistente de modelamiento y estimación · Regresión lineal simple.

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui import components as C  # noqa: E402
from ui import state  # noqa: E402

st.set_page_config(page_title="Asistente de Regresión Lineal Simple", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
C.inject_css()
state.init()

MODULES = [
    ("m01_contexto", "1 · Problema de negocio"),
    ("m02_datos", "2 · Carga y exploración"),
    ("m03_preparacion", "3 · Preparación de datos"),
    ("m04_eda", "4 · Análisis exploratorio"),
    ("m05_y", "5 · Variable a estimar (Y)"),
    ("m06_x", "6 · Variables explicativas (X)"),
    ("m07_modelos", "7 · Modelos de regresión"),
    ("m08_evaluacion", "8 · Evaluación y selección"),
    ("m09_interpretacion", "9 · Interpretación"),
    ("m10_simulacion", "10 · Simulación"),
    ("m11_sensibilidad", "11 · Sensibilidad"),
    ("m12_informe", "12 · Informe ejecutivo"),
    ("m13_aprendizaje", "13 · Centro de aprendizaje"),
]
SIMPLE = [
    ("s1_datos", "1 · Datos"),
    ("s2_variables", "2 · Variables"),
    ("s3_modelo", "3 · Resultado"),
    ("s4_simular", "4 · Simular"),
    ("s5_informe", "5 · Informe"),
    ("m13_aprendizaje", "📚 Aprende"),
]
LABELS = {**dict(MODULES), **dict(SIMPLE)}


def done(key: str) -> bool:
    ss = st.session_state
    return {
        "m01_contexto": ss.ctx.filled_fields() >= 3,
        "m02_datos": state.has_data(),
        "m03_preparacion": len(ss.plan) > 0,
        "m04_eda": "m04_eda" in ss.visited,
        "m05_y": bool(ss.y),
        "m06_x": bool(ss.x_cands),
        "m07_modelos": ss.cmp is not None,
        "m08_evaluacion": bool(ss.chosen),
        "m09_interpretacion": "m09_interpretacion" in ss.visited,
        "m10_simulacion": "m10_simulacion" in ss.visited,
        "m11_sensibilidad": "m11_sensibilidad" in ss.visited,
        "m12_informe": "m12_informe" in ss.visited,
        "m13_aprendizaje": "m13_aprendizaje" in ss.visited,
        "s1_datos": state.has_data(),
        "s2_variables": bool(ss.y) and bool(ss.x_cands) and "s2_variables" in ss.visited,
        "s3_modelo": "s3_modelo" in ss.visited,
        "s4_simular": "s4_simular" in ss.visited,
        "s5_informe": "s5_informe" in ss.visited,
    }[key]


def _switch_mode() -> None:
    lst = MODULES if ss.expert else SIMPLE
    if ss.get("nav") not in dict(lst):
        ss["nav"] = lst[0][0]


ss = st.session_state
ss.setdefault("expert", False)
if "nav" not in ss:
    ss["nav"] = SIMPLE[0][0]
NAV = MODULES if ss.expert else SIMPLE
if ss["nav"] not in dict(NAV):
    ss["nav"] = NAV[0][0]

with st.sidebar:
    st.markdown("### 📈 Regresión lineal simple")
    st.caption("Asistente de ciencia de datos · modelar, estimar y decidir")
    st.radio("Pasos", [k for k, _ in NAV], key="nav",
             format_func=lambda k: f"{'✅' if done(k) else '○'}  {LABELS[k]}", label_visibility="collapsed")
    st.divider()
    st.toggle("Modo experto (13 módulos)", key="expert", on_change=_switch_mode,
              help="Muestra todos los controles: contexto de negocio, limpieza manual, análisis exploratorio, diagnóstico detallado…")
    with st.expander("Opciones"):
        st.selectbox("Nivel de confianza de los intervalos", [90, 95, 99], key="conf", format_func=lambda v: f"{v}%")
        ss.units["y"] = st.text_input("Unidad de Y", ss.units.get("y", ""), placeholder="p. ej. millones de pesos")
        ss.units["x"] = st.text_input("Unidad de X", ss.units.get("x", ""), placeholder="p. ej. km")
    if state.has_data():
        st.caption(f"Datos: {ss.source_name}  \n{ss.loaded.df.shape[0]:,} filas × {ss.loaded.df.shape[1]} columnas")
    if st.button("↺ Reiniciar análisis", help="Borra el plan de limpieza, la selección de variables y los modelos (conserva los datos)."):
        state.reset_analysis()
        st.rerun()
    st.divider()
    st.caption("Las recomendaciones automáticas son heurísticas explicables, no un modelo de lenguaje: la decisión final es suya.")

key = ss["nav"]
ss.visited.add(key)
if key.startswith("s"):
    from ui import simple
    getattr(simple, key)()
else:
    importlib.import_module(f"ui.{key}").render()
