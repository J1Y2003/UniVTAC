"""GR00T N1.7 policy plug-in for UniVTAC's evaluator.

``scripts/eval_policy.py`` does ``importlib.import_module(f"policy.{policy_name}")``
and then instantiates ``policy_module.Policy(deploy_config)``, so ``Policy`` must
be re-exported here.
"""

from .deploy_policy import Policy

__all__ = ["Policy"]
