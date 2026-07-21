from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import yaml

if TYPE_CHECKING:
  from tricolour.cli import FlaggingStrategy


def load_config(config_file) -> Dict[str, Any]:
  """
  Parameters
  ----------
  config_file : str

  Returns
  -------
  dict
    Configuration
  """
  with open(config_file) as cf:
    return yaml.full_load(cf)


def log_configuration(flagging_strategy: FlaggingStrategy, cfg: Dict[str, Any]):
  if len(strategies := cfg.get("strategies", [])) == 0:
    print("Configuration contains no flagging strategies")
    return

  print("*****************************************")
  print("Applying the following strategies:       ")
  print("*****************************************")

  for s, strategy in enumerate(strategies):
    name = strategy.get("name", "<none>")

    if (task := strategy.get("task")) is None:
      print(f"Strategy '{name}' has no associated task")

    print(f"{s}: {task} ({name})")

    for key, value in strategy.get("kwargs", {}).items():
      print(f"\t{key}: {value}", key, value)

  print("***************** END ********************")

  if flagging_strategy == "polarisation":
    print("Flagging based on quadrature polarized power")
  elif flagging_strategy == "total_power":
    print("Flagging on total quadrature power")
  else:
    print("Flagging per correlation ('standard' mode)")
