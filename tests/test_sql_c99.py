"""SQL + C99 parity: generated code == model scoring (tolerance 1e-7)."""
import os
import shutil
import sqlite3
import subprocess
import tempfile

import numpy as np
import pytest

from cardboost.codec.c99_export import to_c99
from cardboost.codec.sql_export import to_sql


def _fit(data, preset="v8"):
    from cardboost.core import engine
    return engine.fit_deploy(data, preset=preset)


@pytest.fixture(scope="module")
def reg_model():
    from cardboost.core.atoms import synthetic_quadratic_2d
    X, y = synthetic_quadratic_2d(n=400, seed=21)
    data = dict(kind="regression", X=X, y=y, names=["u", "v"], geo=None,
                panel_name="sql_reg")
    return _fit(data), X


def test_sql_sqlite_parity(reg_model):
    res, X = reg_model
    sql = to_sql(res.dec_card, table_name="scoring", dialect="sqlite",
                 names=["u", "v"])
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE scoring (u REAL, v REAL)")
    con.executemany("INSERT INTO scoring VALUES (?, ?)",
                    [(float(a), float(b)) for a, b in X])
    got = [r[0] for r in con.execute(
        "SELECT prediction FROM (" + sql + ") ORDER BY card_score")]
    con.close()
    exp = np.asarray(
        __import__("r76_container31", fromlist=["model_score"]).model_score(
            res.dec_card, X), float)
    order = np.argsort(exp)
    assert float(np.max(np.abs(np.asarray(got) - exp[order]))) < 1e-7


def test_sql_all_dialects_emit(reg_model):
    res, X = reg_model
    for dialect in ("postgres", "clickhouse", "snowflake", "bigquery", "sqlite"):
        sql = to_sql(res.dec_card, table_name="t", dialect=dialect,
                     names=["u", "v"])
        assert "SELECT" in sql and "FROM" in sql
        assert "card_score" in sql


def test_sql_handles_null_nan_shelf(reg_model):
    res, X = reg_model
    sql = to_sql(res.dec_card, table_name="scoring", dialect="sqlite",
                 names=["u", "v"])
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE scoring (u REAL, v REAL)")
    con.execute("INSERT INTO scoring VALUES (NULL, 1.0)")
    con.execute("INSERT INTO scoring VALUES (1.0, NULL)")
    got = sorted(r[0] for r in con.execute("SELECT prediction FROM (" + sql + ")"))
    con.close()
    Xnan = np.array([[np.nan, 1.0], [1.0, np.nan]])
    exp = np.asarray(
        __import__("r76_container31", fromlist=["model_score"]).model_score(
            res.dec_card, Xnan), float)
    assert float(np.max(np.abs(np.asarray(got) - np.sort(exp)))) < 1e-7


def test_c99_compiles_and_parity(reg_model):
    gcc = shutil.which("gcc") or shutil.which("cc")
    if gcc is None:
        pytest.skip("no C compiler available")
    res, X = reg_model
    hdr = to_c99(res.dec_card, names=["u", "v"])
    tmp = tempfile.mkdtemp()
    try:
        hpath = os.path.join(tmp, "model.h")
        with open(hpath, "w") as fh:
            fh.write(hdr)
        rows = "\n".join(f"{float(a)!r},{float(b)!r}" for a, b in X)
        driver = f"""
#include <stdio.h>
#include "model.h"
int main(void) {{
    double x[2]; double s;
    while (scanf("%lf,%lf", &x[0], &x[1]) == 2) {{
        s = cardboost_score(x);
        printf("%.17g\\n", s);
    }}
    return 0;
}}
"""
        cpath = os.path.join(tmp, "drv.c")
        with open(cpath, "w") as fh:
            fh.write(driver)
        binp = os.path.join(tmp, "drv")
        comp = subprocess.run([gcc, "-O2", "-o", binp, cpath], cwd=tmp,
                              capture_output=True, text=True)
        assert comp.returncode == 0, comp.stderr[-2000:]
        run = subprocess.run([binp], input=rows, capture_output=True, text=True)
        got = [float(line) for line in run.stdout.split()]
        exp = np.asarray(
            __import__("r76_container31", fromlist=["model_score"]).model_score(
                res.dec_card, X), float)
        assert len(got) == len(exp)
        assert float(np.max(np.abs(np.asarray(got) - exp))) < 1e-7
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scorecard_df_and_local_explain(reg_model):
    res, X = reg_model

    class _M:
        dec_card_ = res.dec_card
        feature_names_ = ["u", "v"]
    from cardboost.explain.local import explain_local
    from cardboost.explain.scorecard_table import scorecard_df
    ex = explain_local(_M(), X[0])
    got = ex["base"] + sum(v for _, v in ex["contributions"])
    exp = float(np.asarray(
        __import__("r76_container31", fromlist=["model_score"]).model_score(
            res.dec_card, X[:1]), float)[0])
    assert abs(got - exp) < 1e-9
    df = scorecard_df(_M(), X=X, y=np.zeros(len(X)))
    assert len(df) == sum(len(c) + 2 for c in res.dec_card.cuts)
