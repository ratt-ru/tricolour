from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import yaml

if TYPE_CHECKING:
  from tricolour.cli import FlaggingStrategy

import logging
log = logging.getLogger("tricolour")

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
    log.info("Configuration contains no flagging strategies")
    return

  log.info("*****************************************")
  log.info("Applying the following strategies:       ")
  log.info("*****************************************")

  for s, strategy in enumerate(strategies):
    name = strategy.get("name", "<none>")

    if (task := strategy.get("task")) is None:
      log.info(f"Strategy '{name}' has no associated task")

    log.info(f"{s}: {task} ({name})")

    for key, value in strategy.get("kwargs", {}).items():
      log.info(f"\t{key}: {value}")

  log.info("***************** END ********************")

  if flagging_strategy == "polarisation":
    log.info("Flagging based on quadrature polarized power")
  elif flagging_strategy == "total_power":
    log.info("Flagging on total quadrature power")
  else:
    log.info("Flagging per correlation ('standard' mode)")
