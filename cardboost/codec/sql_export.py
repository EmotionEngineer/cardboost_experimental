"""Exact SQL export (postgres / clickhouse / snowflake / bigquery / sqlite).

Generates a pure-SQL scoring query from a fitted card IR.  Routing matches the
engine ``bin_idx`` bit-for-bit within ``1e-7``: right-open intervals
``(-inf, c0), [c0, c1), ..., [c_last, +inf)`` plus the NaN shelf as a dedicated
branch (NULLs fall through SQL comparisons, so the NULL branch goes first).
"""
from __future__ import annotations

from typing import List, Optional

from ._sem import CardIR, IsoIR, _bin_sem

_DIALECTS = {
    "postgres": ('"', '"'),
    "sqlite": ('"', '"'),
    "snowflake": ('"', '"'),
    "bigquery": ("`", "`"),
    "clickhouse": ("`", "`"),
}


def _q(name: str, dialect: str) -> str:
    o, c = _DIALECTS[dialect]
    return f"{o}{name}{c}"


def _num(x: float) -> str:
    return repr(float(x))


def _bin_case(var: str, cuts: List[float], vals: List[str], nan_val: str) -> str:
    """CASE routing identical to bin_idx (NaN shelf last bin)."""
    parts = [f"CASE WHEN {var} IS NULL THEN {nan_val}"]
    for i, cut in enumerate(cuts):
        parts.append(f"WHEN {var} < {_num(cut)} THEN {vals[i]}")
    parts.append(f"ELSE {vals[len(cuts)]} END")
    return " ".join(parts)


def _vals_levels(levels, dim: int, cls: int) -> List[str]:
    if dim > 1:
        return [_num(levels[i][cls]) for i in range(len(levels))]
    return [_num(levels[i]) for i in range(len(levels))]


def _interp_case(var: str, kx: List[float], kv: List[float]) -> str:
    """Clamped piecewise-linear interpolation (np.interp semantics)."""
    n = len(kx)
    if n == 1:
        return _num(kv[0])
    segs = []
    for s in range(n - 1):
        dx = kx[s + 1] - kx[s]
        slope = (kv[s + 1] - kv[s]) / dx if dx != 0 else 0.0
        segs.append(f"WHEN {var} < {_num(kx[s + 1])} THEN "
                    f"({_num(kv[s])} + ({var} - {_num(kx[s])}) * {_num(slope)})")
    return (f"CASE WHEN {var} <= {_num(kx[0])} THEN {_num(kv[0])} "
            + " ".join(segs)
            + f" ELSE {_num(kv[-1])} END")


def _rank(var: str, g: List[float], r: List[float]) -> str:
    return _interp_case(var, g, r)


def score_expr(ir: CardIR, cls: int = 0, names: Optional[List[str]] = None) -> str:
    """Additive score expression (before IsoLink) for class `cls`."""
    names = names or ir.names or [f"x{j}" for j in range(len(ir.mains))]
    dim = ir.n_classes
    terms: List[str] = [_num(ir.b0[cls]) if dim > 1 else _num(ir.b0)]

    for m in ir.mains:
        var = _q(names[m.j], _DIALECT_DEFAULT)
        vals = _vals_levels(m.levels, dim, cls)
        terms.append(f"({_bin_case(var, m.cuts, vals, vals[-1])})")

    for a in ir.rays:
        xj, xk = _q(names[a.j], _DIALECT_DEFAULT), _q(names[a.k], _DIALECT_DEFAULT)
        rj = _rank(xj, a.gj, a.rgj)
        rk = _rank(xk, a.gk, a.rgk)
        u = f"({_num(a.a)} * ({rj}) + {_num(a.b)} * ({rk}))"
        vals = _vals_levels(a.levels, dim, cls)
        ub = _bin_case(u, a.ucuts, vals, vals[-1])
        ins = (f"({xj} >= {_num(a.xmin_j)} AND {xj} <= {_num(a.xmax_j)} "
               f"AND {xk} >= {_num(a.xmin_k)} AND {xk} <= {_num(a.xmax_k)})")
        terms.append(f"(CASE WHEN {ins} THEN ({ub}) ELSE 0.0 END)")

    for g in ir.grids:
        xj, xk = _q(names[g.j], _DIALECT_DEFAULT), _q(names[g.k], _DIALECT_DEFAULT)
        A, B = len(g.cj) + 2, len(g.ck) + 2
        rows = []
        for a in range(A):
            vals = [_num(g.table[a][b][cls] if dim > 1 else g.table[a][b])
                    for b in range(B)]
            rows.append(_bin_case(xk, g.ck, vals, vals[-1]))
        inner = _bin_case(xj, g.cj, rows, rows[-1])
        ins = (f"({xj} >= {_num(g.xmin_j)} AND {xj} <= {_num(g.xmax_j)} "
               f"AND {xk} >= {_num(g.xmin_k)} AND {xk} <= {_num(g.xmax_k)})")
        terms.append(f"(CASE WHEN {ins} THEN ({inner}) ELSE 0.0 END)")

    for c in ir.cells:
        xj, xk = _q(names[c.j], _DIALECT_DEFAULT), _q(names[c.k], _DIALECT_DEFAULT)
        A, B = len(c.cj) + 2, len(c.ck) + 2
        rows = []
        for a in range(A):
            vals = [_num(c.table[a][b][cls] if dim > 1 else c.table[a][b])
                    for b in range(B)]
            rows.append(_bin_case(xk, c.ck, vals, vals[-1]))
        terms.append(f"({_bin_case(xj, c.cj, rows, rows[-1])})")

    for q in ir.qmiss:
        var_names = [_q(names[j], _DIALECT_DEFAULT) for j, _ in q.shelves]
        qsum = " + ".join(
            f"(CASE WHEN {v} IS NULL THEN {_num(w)} ELSE 0.0 END)"
            for (j, w), v in zip(q.shelves, var_names)) or "0.0"
        vals = _vals_levels(q.levels, dim, cls)
        terms.append(f"({_bin_case('(' + qsum + ')', q.qcuts, vals, vals[-1])})")

    return " + ".join(f"({t})" for t in terms)


_DIALECT_DEFAULT = "postgres"


def to_sql(card, table_name: str = "scoring_view", dialect: str = "postgres",
           names: Optional[List[str]] = None, score_col: str = "card_score",
           pred_col: str = "prediction") -> str:
    """Return the full scoring SELECT for `card` against `table_name`."""
    from ._sem import card_to_ir
    global _DIALECT_DEFAULT
    if dialect not in _DIALECTS:
        raise ValueError(f"unknown dialect {dialect!r}; use one of {sorted(_DIALECTS)}")
    ir = card if isinstance(card, CardIR) else card_to_ir(card, names=names)
    _DIALECT_DEFAULT = dialect
    try:
        names = names or ir.names or [f"x{j}" for j in range(len(ir.mains))]
        header = [
            "-- generated by cardboost 1.0.0 (Container v3.x, bit-verified)",
            f"-- routing: right-open intervals + NaN shelf; dialect: {dialect}",
            "SELECT",
        ]
        if ir.n_classes > 1:
            cols = []
            for c in range(ir.n_classes):
                cols.append(f"  {score_expr(ir, c, names)} AS {_q(score_col + f'_c{c}', dialect)},")
            body = cols
            exprs = [f"{_q(score_col + f'_c{c}', dialect)}" for c in range(ir.n_classes)]
            best = exprs[0]
            for e in exprs[1:]:
                best = f"CASE WHEN {e} > {best} THEN {e} ELSE {best} END"
            body.append(f"  {best} AS {_q(score_col, dialect)},")
            body.append(f"  CASE " + " ".join(
                f"WHEN {_q(score_col + f'_c{c}', dialect)} >= {best} THEN {c} "
                for c in range(ir.n_classes)) + f"ELSE 0 END AS {_q(pred_col, dialect)}")
        else:
            raw = score_expr(ir, 0, names)
            if ir.iso is not None:
                pred = _interp_case(f"({raw})", ir.iso.kx, ir.iso.kv)
            else:
                pred = f"({raw})"
            body = [
                f"  {raw} AS {_q(score_col, dialect)},",
                f"  {pred} AS {_q(pred_col, dialect)}",
            ]
        tail = f"FROM {_q(table_name, dialect)}"
        return "\n".join(header + body + [tail])
    finally:
        _DIALECT_DEFAULT = "postgres"


def export_table(card, names: Optional[List[str]] = None):
    """Row-oriented scorecard export of the MAINS (Feature, cut, level).

    Pair atoms and IsoLink are reported separately as dictionaries.
    """
    from ._sem import card_to_ir
    ir = card if isinstance(card, CardIR) else card_to_ir(card, names=names)
    rows = []
    for m in ir.mains:
        edges = [float("-inf")] + m.cuts + [float("nan")]
        for b, lvl in enumerate(m.levels):
            rows.append((m.name, b, edges[b] if b < len(edges) - 1 else None,
                         edges[b + 1] if b + 1 < len(edges) else None,
                         lvl if not isinstance(lvl, list) else list(lvl)))
    return rows
