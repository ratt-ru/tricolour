from dataclasses import dataclass
from typing import Dict, Literal, TypeAlias

NodeId: TypeAlias = str
Action: TypeAlias = Literal["read", "flag", "write"]
FlagStrategy: TypeAlias = Literal["standard", "total_power", "polarisation"]


@dataclass(slots=True)
class WorkItem:
  path: str
  region: Dict[str, slice]

  def __hash__(self):
    return hash((self.path, frozenset(self.region.items())))
