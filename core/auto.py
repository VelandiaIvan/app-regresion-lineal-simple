"""Modo automático: detecta la variable a estimar, limpia los datos y elige variables explicativas sin pedir nada al usuario.

Todo es explicable: cada decisión devuelve una frase en lenguaje sencillo para mostrarla en pantalla.
"""
from __future__ import annotations

import re

import pandas as pd

from . import cleaning, eda, profiling
from .cleaning import CleaningPlan
from .context import BusinessContext, rank_predictor_candidates, rank_target_candidates


def suggest_target(df: pd.DataFrame, objetivo: str = "", ctx: BusinessContext | None = None) -> tuple[str | None, pd.DataFrame]:
    """Variable Y sugerida. Usa el texto del usuario si lo hay; si no, solo los datos."""
    ctx = ctx or BusinessContext()
    if objetivo.strip():
        ctx = BusinessContext(**{**ctx.to_dict(), "objetivo_y": objetivo})
    rank = rank_target_candidates(df, profiling.classify_columns(df), ctx)
    if not len(rank):
        return None, rank
    best = rank.iloc[0]["variable"]
    # «Gemelos»: dos columnas casi idénticas (r ≥ 0,98). Si la elegida tiene nombre de medida derivada
    # (neto, desembolsado, facturado…), se prefiere la original.
    if len(rank) > 1 and not ctx.objetivo_y.strip():
        other = rank.iloc[1]["variable"]
        r = df[[best, other]].corr().iloc[0, 1]
        if abs(r) >= 0.98 and _DERIVED.search(best.lower()) and not _DERIVED.search(other.lower()):
            best = other
            rank = pd.concat([rank.iloc[[1]], rank.iloc[[0]], rank.iloc[2:]], ignore_index=True)
    return best, rank


_DERIVED = re.compile(r"neto|netos|desembols|facturad|ajustad|deflactad|acumulad")


def auto_plan(df: pd.DataFrame, y: str | None = None, date_col: str | None = None) -> tuple[CleaningPlan, list[str]]:
    """Plan de limpieza recomendado + explicación de cada decisión."""
    plan, why = CleaningPlan(), []
    dup = int(df.duplicated().sum())
    if dup:
        plan.add("drop_duplicates", None, "drop_duplicates")
        why.append(f"Se eliminan {dup} filas repetidas.")
    inc = profiling.find_inconsistencies(df)
    for _, r in inc.iterrows():
        v, p = r["variable"], r["problema"]
        if p == "negativos improbables":
            plan.add("fix_nan", v, "a_faltante", rule="negativos")
            why.append(f"«{v}»: {r['n_afectados']} valores negativos imposibles se tratan como dato faltante.")
        elif p == "categorías duplicadas":
            plan.add("normalize_labels", v, "normalizar")
            why.append(f"«{v}»: se unifican categorías escritas de forma distinta (mayúsculas, tildes, espacios).")
        elif p == "posible código de dato faltante":
            m = re.search(r"valor (-?\d+)", r["detalle"])
            if m:
                plan.add("fix_nan", v, "a_faltante", rule="valor", value=float(m.group(1)))
                why.append(f"«{v}»: el código {m.group(1)} se trata como dato faltante.")
    # faltantes (después de las correcciones anteriores, que generan nuevos faltantes)
    work = cleaning.apply_plan(df, plan, protect=[y] if y else []).df
    for c in work.columns:
        if c == y or work[c].isna().sum() == 0:
            continue
        adv = cleaning.recommend_imputation(work, c, ordered_by=date_col)
        if adv.recommended in {"mean", "median", "mode", "ffill", "interpolate", "drop_rows"}:
            plan.add("impute", c, adv.recommended)
        elif adv.recommended == "constant":
            plan.add("impute", c, "constant", value=adv.params.get("value", "Sin dato"))
        elif adv.recommended == "drop_column":
            plan.add("drop_column", c, "drop_column")
        else:
            continue
        why.append(f"«{c}»: {adv.pct_missing:.0f}% faltante → {cleaning.METHOD_LABELS.get(adv.recommended, adv.recommended).lower()}.")
    if y and df[y].isna().any():
        why.append(f"«{y}» (lo que se estima) tiene {int(df[y].isna().sum())} faltantes: esas filas se descartan, nunca se inventan.")
    if not why:
        why.append("Los datos no necesitan limpieza.")
    return plan, why


def suggest_predictors(md, ctx: BusinessContext | None = None, k: int = 4) -> tuple[list[str], pd.DataFrame]:
    """Hasta k variables X (solo con datos de entrenamiento), excluyendo identificadores y posible fuga."""
    ctx = ctx or BusinessContext()
    train = md.df.loc[md.train_idx]
    cls = profiling.classify_columns(md.df)
    drop = {md.y_source, md.y_col} | {c for c, t in md.transforms.items() if t["source"] == md.y_source}
    cands = [c for c in cls[cls["usable_como_X"]]["variable"] if c not in drop
             and pd.api.types.is_numeric_dtype(md.df[c]) and not str(c).endswith("__atípico")]
    if not cands:
        return [], pd.DataFrame()
    corr = eda.correlation_table(train, md.y_col, cands)
    rank = rank_predictor_candidates(train, md.y_source, cls, ctx, corr)
    if rank.empty:
        return [], rank
    ok = rank[(rank["riesgo_fuga"] != "alta") & (rank["p_valor"] < 0.10)]
    pick = ok.head(k)["variable"].tolist() or rank[rank["riesgo_fuga"] != "alta"].head(2)["variable"].tolist()
    return pick, rank
