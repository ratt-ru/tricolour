"""FastAPI surface for the progress dashboard.

Ray Serve already runs an HTTP proxy for the application, so exposing a
dashboard is a matter of mounting routes on the driver deployment rather than
standing up a second server. The routes live in the deployment class body
(Serve's ``@serve.ingress`` contract requires that) and delegate straight to
:class:`~tricolour.core.application.progress.RunState`, which is where the
behaviour worth testing lives.
"""

from __future__ import annotations

import asyncio
import json
from importlib.resources import files as resource_files
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import FastAPI

if TYPE_CHECKING:
  from tricolour.core.application.progress import RunState

#: Interval between server-sent snapshots. The payload is small and the page
#: only redraws what changed, so a second is comfortable.
SNAPSHOT_INTERVAL_S = 1.0

dashboard_app = FastAPI(title="tricolour", docs_url=None, redoc_url=None, openapi_url=None)


def index_html() -> str:
  """The single-page dashboard.

  Read on demand rather than at import so that editing the page during
  development does not require restarting a cluster.
  """
  return (resource_files("tricolour") / "static" / "index.html").read_text(encoding="utf-8")


async def snapshot_stream(state: RunState, interval: float = SNAPSHOT_INTERVAL_S) -> AsyncGenerator[str, None]:
  """Yield SSE frames carrying run snapshots until the run finishes.

  One final frame is emitted after completion so that a page connecting late,
  or still open when the run ends, settles on the finished state.
  """
  try:
    while True:
      finished = state.finished_at is not None
      yield f"data: {json.dumps(state.snapshot(), default=str)}\n\n"

      if finished:
        return

      await asyncio.sleep(interval)
  except asyncio.CancelledError:
    # The browser navigated away or reloaded; nothing to clean up.
    raise
