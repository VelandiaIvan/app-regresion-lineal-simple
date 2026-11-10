"""Modo sencillo: 5 pasos con detección y sugerencias automáticas (pensado para uso académico)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import auto, interpretation as itp, loader
from core import simulation as sim_
from core.cleaning import CleaningPlan
from core.diagnostics import ICON, LABEL
from core.report import build_executive_report
from sample_data import generate as G
from ui import charts
from ui import components as C
from ui import state
from ui.m07_modelos import build, model_points

STATUS = {"recomendado": ("ok", "✅ Modelo confiable"), "con reservas": ("warn", "⚠️ Modelo útil, con reservas"),
          "no confiable": ("bad", "⛔ No hay un modelo lineal confiable"), "sin candidatos": ("bad", "⛔ Sin modelo")}


# ============================================================================ ayudas de estado
def _ensure_auto() -> None:
    """Aplica Y, limpieza y X automáticas cuando cambian los datos o la Y elegida."""
    ss = st.session_state
    if not state.has_data():
        return
    if ss.get("_auto_data") != ss.data_id:
        y, _ = auto.suggest_target(ss.loaded.df, ctx=ss.ctx)
        ss.y, ss.y_transform = y, None
        ss._auto_data, ss._auto_y = ss.data_id, None
    if ss.y and ss.get("_auto_y") != (ss.y, ss.get("auto_clean", True)):
        if ss.get("auto_clean", True):
            ss.plan, ss.auto_why = auto.auto_plan(ss.loaded.df, ss.y, ss.date_col)
        else:
            ss.plan, ss.auto_why = CleaningPlan(), ["Limpieza automática desactivada: se usan los datos tal como llegaron."]
        md, _ = state.modeling()
        ss.x_cands = auto.suggest_predictors(md, ss.ctx)[0] if md is not None else []
        ss.x_meta, ss.leak_cleared = {}, set()
        ss._auto_y = (ss.y, ss.get("auto_clean", True))


def _need_data() -> None:
    if not state.has_data():
        C.callout("Primero elija un ejemplo o cargue un archivo en el paso 1.", "info")
        st.button("Ir al paso 1 ▶", on_click=C.goto, args=("s1_datos",))
        st.stop()


def _model():
    """Construye (si hace falta) y devuelve (comparación, modelo elegido)."""
    ss = st.session_state
    _ensure_auto()
    if not ss.x_cands:
        C.callout("No se encontraron variables numéricas que puedan explicar la variable elegida. Pruebe con otra en el paso 2.", "bad")
        st.stop()
    md, err = state.modeling()
    if md is None:
        st.error(err)
        st.stop()
    if not state.cmp_is_current():
        with st.spinner("Calculando modelos…"):
            build()
    cmp = ss.cmp
    if not cmp.models:
        st.error("No se pudo construir ningún modelo con estos datos.")
        st.stop()
    if ss.chosen not in cmp.models:
        ss.chosen = cmp.recommendation.best_key
    return cmp, cmp.models[ss.chosen]


# ============================================================================ paso 1
def _load_example() -> None:
    state.load_sample(st.session_state["sample_sel"])


def s1_datos() -> None:
    ss = st.session_state
    C.hero("Paso 1 de 5", "Sus datos", "Elija un ejemplo o cargue su archivo. La aplicación hace el resto: detecta qué estimar, limpia los datos y sugiere las variables.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🧪 Usar un ejemplo")
        key = st.selectbox("Caso de ejemplo", list(G.DATASETS), key="sample_sel", label_visibility="collapsed")
        st.caption(G.DATASETS[key][2])
        st.button("Cargar ejemplo", on_click=_load_example, type="primary")
    with c2:
        st.markdown("##### 📂 O cargar su archivo")
        up = st.file_uploader("CSV, TXT, XLS o XLSX", type=["csv", "txt", "tsv", "xls", "xlsx", "xlsm"], key="s_uploader")
        if up is not None and ss.get("_file_sig") != (up.name, up.size):
            try:
                state.set_loaded(loader.load_file(up, up.name), up.name)
                ss["_file_sig"] = (up.name, up.size)
                ss.ctx = type(ss.ctx)()          # archivo propio: sin contexto de ejemplo
            except Exception as exc:
                st.error(f"No se pudo leer el archivo: {exc}")
    if not state.has_data():
        return
    df = ss.loaded.df
    st.divider()
    st.markdown(f"#### {ss.source_name}")
    C.kpis([("Registros", f"{len(df):,}"), ("Columnas", f"{df.shape[1]}"),
            ("Datos faltantes", f"{df.isna().mean().mean():.1%}"), ("Filas repetidas", f"{int(df.duplicated().sum())}")])
    st.dataframe(df.head(8), hide_index=True)
    _ensure_auto()
    with st.expander("🧹 Lo que la aplicación revisó y corrigió automáticamente", expanded=True):
        for w in ss.get("auto_why", []):
            st.markdown(f"- {w}")
    C.next_step("Siguiente: variables ▶", "s2_variables")


# ============================================================================ paso 2
def s2_variables() -> None:
    ss = st.session_state
    C.hero("Paso 2 de 5", "¿Qué estimar y con qué?", "La aplicación ya eligió por usted. Revise y cambie solo si quiere.")
    _need_data()
    _ensure_auto()
    df = ss.loaded.df
    y_auto, rank = auto.suggest_target(df, ctx=ss.ctx)
    opts = rank["variable"].tolist()
    if not opts:
        st.error("El archivo no tiene columnas numéricas que se puedan estimar.")
        st.stop()
    st.markdown("#### 🎯 Variable a estimar (Y)")
    top = rank.set_index("variable")
    y = st.selectbox("Variable a estimar", opts, index=opts.index(ss.y) if ss.y in opts else 0, label_visibility="collapsed")
    if y != ss.y:
        ss.y, ss.y_transform = y, None
        _ensure_auto()
        st.rerun()
    why = top.loc[y, "razones"] if y in top.index else ""
    C.callout(f"<b>{y}</b>{' (sugerida)' if y == y_auto else ''}: {why}.", "info")

    st.markdown("#### 🔎 Variables que pueden explicarla (X)")
    md, err = state.modeling()
    if md is None:
        st.error(err)
        st.stop()
    _, xr = auto.suggest_predictors(md, ss.ctx)
    allx = xr["variable"].tolist() if len(xr) else []
    sel = st.multiselect("Se construirá un modelo con cada una", allx, default=[x for x in ss.x_cands if x in allx],
                         label_visibility="collapsed")
    if sel != ss.x_cands:
        ss.x_cands = sel
    if len(xr):
        view = xr[["variable", "pearson_r", "riesgo_fuga", "a_favor", "alertas"]].rename(columns={
            "pearson_r": "correlación", "riesgo_fuga": "riesgo de trampa", "a_favor": "por qué sirve", "alertas": "cuidado"})
        C.table(view, 2)
        leaks = xr[xr["riesgo_fuga"] == "alta"]["variable"].tolist()
        if leaks:
            C.callout("No se sugieren <b>" + ", ".join(leaks) + "</b>: se parecen demasiado a Y (probablemente se calculan a partir de ella), "
                      "así que «predecir» con ellas sería hacer trampa.", "warn")
    with st.expander("⚙️ Opciones (no es necesario tocarlas)"):
        ss.setdefault("auto_clean", True)
        st.toggle("Limpieza automática de datos", key="auto_clean",
                  help="Elimina repetidos, corrige valores imposibles y completa datos faltantes.")
        ss.y_transform = {"No": None, "Sí (logaritmo)": "ln"}[st.radio(
            "¿Usar el logaritmo de Y? (útil si Y tiene valores muy grandes y dispersos)", ["No", "Sí (logaritmo)"],
            index=1 if ss.y_transform == "ln" else 0, horizontal=True,
            disabled=not bool((df[ss.y].dropna() > 0).all()))]
    if not ss.x_cands:
        st.warning("Elija al menos una variable X.")
    else:
        C.next_step("Siguiente: ver el resultado ▶", "s3_modelo")


# ============================================================================ paso 3
def s3_modelo() -> None:
    ss = st.session_state
    C.hero("Paso 3 de 5", "Resultado", "Se probó un modelo por cada variable y se eligió el mejor con datos que el modelo no había visto.")
    _need_data()
    cmp, cm = _model()
    rec, res = cmp.recommendation, cm.res
    kind, title = STATUS[rec.status]
    msg = {"recomendado": "Puede usarlo para estimar.", "con reservas": "Sirve para estimar, pero lea las advertencias.",
           "no confiable": "La relación es débil o no es una línea recta: tómelo solo como referencia."}.get(rec.status, "")
    C.callout(f"Mejor variable para explicar <b>{res.y_source}</b>: <b>{cm.key}</b>. {msg}", kind, title)
    st.markdown(f"<div class='eq'>{res.equation()}</div>", unsafe_allow_html=True)
    C.callout(itp.plain_language(res, ss.units.get("x", ""), ss.units.get("y", "")), "edu", "Interpretación")
    tm = cm.test_metrics or {}
    C.kpis([("R² (qué tanto explica)", f"{res.r2:.0%}"),
            ("Error típico", f"±{res.rmse:,.4g}", "en unidades de Y"),
            ("Acierto con datos nuevos", f"{tm['R²_fuera_muestra']:.0%}" if tm else "—", "R² en prueba"),
            ("¿Es casualidad?", "No" if res.p_b1 < res.alpha else "Posiblemente", f"p = {res.p_b1:.2g}")])
    x_t, y_t = model_points(cm, md=cmp.md)
    C.chart(charts.regression_chart(res, x_t, y_t), "s3_reg")
    if rec.warnings:
        with st.expander("⚠️ Advertencias", expanded=rec.status != "recomendado"):
            for w in rec.warnings:
                st.markdown(f"- {w}")
            for a in rec.alternatives:
                st.markdown(f"- 💡 {a}")
    t1, t2, t3 = st.tabs(["Comparar modelos", "¿Se cumplen los supuestos?", "Detalle técnico"])
    with t1:
        keys = list(cmp.models)
        ss.chosen = st.selectbox("Modelo a usar en los pasos siguientes", keys, index=keys.index(ss.chosen),
                                 format_func=lambda k: f"{k}{'  ★ recomendado' if k == rec.best_key else ''}")
        C.table(cmp.table[["ranking", "modelo", "puntaje", "R²", "R²_prueba", "p_β1", "alertas"]], 3)
        if rec.note_r2:
            st.caption(rec.note_r2)
    with t2:
        d = cm.diag
        cols = st.columns(len(d.checks))
        for c, chk in zip(cols, d.checks):
            c.markdown(f"**{chk.name}**<br>{ICON[chk.severity]} {LABEL[chk.severity]}", unsafe_allow_html=True)
        st.caption(d.summary)
        g = st.columns(2)
        with g[0]:
            C.chart(charts.residuals_vs_fitted(res), "s3_rvf")
        with g[1]:
            C.chart(charts.qq_chart(res), "s3_qq")
    with t3:
        C.table(res.coef_table(), 4)
        for t in itp.explain_metrics(res, cm.test_metrics):
            st.markdown(f"- {t}")
    C.next_step("Siguiente: simular ▶", "s4_simular")


# ============================================================================ paso 4
def s4_simular() -> None:
    ss = st.session_state
    C.hero("Paso 4 de 5", "¿Qué pasa si…?", "Mueva X y vea cómo cambia la estimación de Y.")
    _need_data()
    cmp, cm = _model()
    res = cm.res
    xl, yl = res.x_source, res.y_source
    lo, hi = res.x_min, res.x_max
    span = hi - lo
    a, b = lo - 0.3 * span, hi + 0.3 * span
    if res.x_transform:
        a = max(a, lo * 0.5 if lo > 0 else 1e-6)
    base = float(np.median(res.x_obs))
    x_new = st.slider(f"Valor de {xl}", float(a), float(b), float(np.clip(base * 1.1, a, b)), step=float((b - a) / 300),
                      key=f"s4_x_{cm.key}", format="%.4g")
    sim = sim_.simulate(res, base, x_new)
    ss["last_sim"], ss["last_sim_key"] = sim, cm.key
    ex = sim["extrap_new"]
    (st.success if ex.status == "dentro" else st.warning if ex.status == "leve" else st.error)(ex.message)
    C.kpis([(f"{yl} con X típico ({base:,.4g})", f"{sim['y_base']:,.4g}"), (f"{yl} con X = {x_new:,.4g}", f"{sim['y_new']:,.4g}"),
            ("Cambio", f"{sim['dy']:+,.4g}", f"{sim['dy_pct']:+.1f}%" if not np.isnan(sim["dy_pct"]) else ""),
            ("Rango probable de un caso", f"{sim['new_pi'][0]:,.4g} – {sim['new_pi'][1]:,.4g}")])
    C.callout(sim_.interpret_simulation(res, sim, xl, yl), "info")
    scen = sim_.scenario_values(res, ss.ctx.y_higher_is_better)
    ss["last_scen"], ss["last_scen_key"] = scen, cm.key
    C.chart(charts.scenario_chart(res, scen, sim), "s4_chart")
    st.markdown("#### Si X sube o baja un porcentaje")
    tab = sim_.sensitivity_table(res, base, x_label=xl, y_label=yl)
    ss["last_sens"], ss["last_sens_key"] = tab, cm.key
    st.dataframe(tab[["escenario", "x_nuevo", "y_nuevo", "dif_absoluta", "var_%"]].rename(columns={
        "escenario": "cambio de X", "x_nuevo": "X", "y_nuevo": "Y estimado", "dif_absoluta": "cambio de Y", "var_%": "cambio %"}),
        hide_index=True, column_config={"cambio %": st.column_config.NumberColumn(format="%+.1f")})
    st.caption("Es una asociación observada en los datos: no prueba que cambiar X cause el cambio en Y.")
    C.next_step("Siguiente: informe ▶", "s5_informe")


# ============================================================================ paso 5
def s5_informe() -> None:
    ss = st.session_state
    C.hero("Paso 5 de 5", "Informe", "Resumen listo para descargar o imprimir.")
    _need_data()
    cmp, cm = _model()
    res = cm.res
    scen = ss.get("last_scen") if ss.get("last_scen_key") == cm.key else sim_.scenario_values(res, ss.ctx.y_higher_is_better)
    sim = ss.get("last_sim") if ss.get("last_sim_key") == cm.key else None
    sens = ss.get("last_sens") if ss.get("last_sens_key") == cm.key else sim_.sensitivity_table(res, scen["base"])
    rep = build_executive_report(ctx=ss.ctx, source_name=ss.source_name, shape_raw=ss.loaded.df.shape, plan_records=ss.plan.records(),
                                 change_log=cmp.md.log, corr_table=None, cmp=cmp, chosen_key=cm.key, units=ss.units,
                                 sim=sim, sens=sens, scenarios=scen)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Descargar informe (HTML)", rep.to_html().encode("utf-8"), "informe_regresion.html", "text/html", type="primary")
    c2.download_button("⬇ Descargar informe (Markdown)", rep.to_markdown().encode("utf-8"), "informe_regresion.md", "text/markdown")
    st.markdown(rep.to_markdown())
