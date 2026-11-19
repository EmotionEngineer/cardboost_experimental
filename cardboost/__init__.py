"""cardboost - Rate-Distortion Optimal Glass-Box Scorecards.

Ultra-compact (<= 4 KB) piecewise-constant additive models with bit-exact
SQL, C99 and binary deployment.  Default preset ``"v8"`` is the frozen R87
champion ``card_v8_base``; see ``cardboost.presets`` for the full arsenal.

Quick start
-----------
>>> from cardboost import CardBoostRegressor
>>> model = CardBoostRegressor(preset="v8").fit(X_train, y_train)
>>> y_hat = model.predict(X_test)
>>> sql = model.to_sql(table_name="scoring", dialect="postgres")
>>> blob = model.to_bytes()
"""
from .estimators import CardBoostClassifier, CardBoostRegressor
from .presets import PRESET_NAMES, get_preset
from .codec.container_v32 import from_bytes
from .codec._sem import UnsupportedAtomError

__version__ = "1.0.0"

__all__ = [
    "CardBoostRegressor",
    "CardBoostClassifier",
    "PRESET_NAMES",
    "get_preset",
    "from_bytes",
    "UnsupportedAtomError",
    "__version__",
]
