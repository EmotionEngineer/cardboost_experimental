"""Explainability: ANOVA-style importance, local decompositions, scorecard tables, OOD audit."""
from .anova import anova_importance
from .local import explain_local
from .scorecard_table import scorecard_df
from .audit import audit_ood_safety

__all__ = ["anova_importance", "explain_local", "scorecard_df", "audit_ood_safety"]
