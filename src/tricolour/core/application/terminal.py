"""Terminal rendering of a run snapshot.

A third renderer over the payload :meth:`RunState.snapshot` already serves to
the dashboard, alongside the web page and anything else that speaks to
``/api/state``. The snapshot is the contract; nothing here reaches into
:class:`RunState` itself, so a progress line can be driven equally well by a
deployment handle in-process or by an HTTP poll from another machine.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any, Dict, TextIO

#: Below this many cells a bar conveys less than the numbers beside it, so the
#: renderer drops it and spends the width on text instead.
MIN_BAR_CELLS = 8

#: The bar takes a fixed share of the line rather than whatever is left over.
#: Letting it absorb the slack means it grows and shrinks as `eta` or the
#: in-flight count come and go, which on a terminal being repainted several
#: times a second reads as the bar jumping about.
MAX_BAR_CELLS = 28

DEFAULT_WIDTH = 80

#: Refresh cadence when a human is watching, versus when the output is a log
#: file or a pipeline's captured stdout.
INTERACTIVE_INTERVAL_S = 0.25
LOG_INTERVAL_S = 30.0


@dataclass(frozen=True)
class BarGlyphs:
  full: str
  empty: str
  partials: str = ""


UNICODE_GLYPHS = BarGlyphs(full="█", empty="░", partials="▏▎▍▌▋▊▉")
ASCII_GLYPHS = BarGlyphs(full="#", empty="-")


def glyphs_for(stream: TextIO) -> BarGlyphs:
  """Pick a glyph set the stream can actually encode.

  A container with a C locale will raise ``UnicodeEncodeError`` part-way
  through a write rather than degrading, which would take the run down with
  it over a progress bar.
  """
  encoding = getattr(stream, "encoding", None)

  if not encoding:
    return ASCII_GLYPHS

  try:
    UNICODE_GLYPHS.full.encode(encoding)
    UNICODE_GLYPHS.empty.encode(encoding)
    UNICODE_GLYPHS.partials.encode(encoding)
  except (UnicodeEncodeError, LookupError):
    return ASCII_GLYPHS

  return UNICODE_GLYPHS


def format_duration(seconds: float | None) -> str:
  """Render a duration the way the run's own log does."""
  if seconds is None or seconds != seconds or seconds == float("inf"):
    return "--"

  seconds = max(0, int(round(seconds)))
  hours, remainder = divmod(seconds, 3600)
  minutes, secs = divmod(remainder, 60)

  if hours:
    return f"{hours}h{minutes:02d}m"
  if minutes:
    return f"{minutes}m{secs:02d}s"

  return f"{secs}s"


def render_bar(fraction: float, cells: int, glyphs: BarGlyphs) -> str:
  """A progress bar of exactly ``cells`` characters."""
  if cells <= 0:
    return ""

  fraction = min(max(fraction, 0.0), 1.0)
  filled = fraction * cells
  whole = int(filled)
  bar = glyphs.full * whole

  # A partial block gives a 50-cell bar the resolution of a 400-cell one,
  # which is the difference between a bar that visibly moves on a long run
  # and one that appears stuck.
  if whole < cells and glyphs.partials:
    if index := int((filled - whole) * (len(glyphs.partials) + 1)):
      bar += glyphs.partials[index - 1]
      whole += 1

  return bar + glyphs.empty * (cells - whole)


def describe_status(snapshot: Dict[str, Any]) -> str:
  if snapshot.get("errors"):
    return "failed"
  if snapshot.get("finished_at"):
    return "done"
  if not snapshot.get("running"):
    return "waiting"
  # `plan` runs after the clock starts, so a run briefly knows it is going
  # without yet knowing how much there is to do.
  return "flagging" if snapshot["progress"]["total_items"] else "planning"


def format_progress(snapshot: Dict[str, Any], width: int = DEFAULT_WIDTH, glyphs: BarGlyphs = UNICODE_GLYPHS) -> str:
  """Render a snapshot as a single line of at most ``width`` characters.

  Fields are dropped from the right as the terminal narrows, so the line
  degrades to a bar and a percentage rather than wrapping.
  """
  progress = snapshot["progress"]
  total = progress["total_items"]
  completed = progress["completed"]
  status = describe_status(snapshot)

  if not total:
    elapsed = format_duration(snapshot.get("elapsed"))
    return f"{status} {elapsed}"[:width]

  fraction = progress["fraction"]
  head = f"{status} "
  percent = f"{fraction * 100:3.0f}%"

  # In priority order; the line keeps as many as fit.
  fields = [f"{completed}/{total}"]

  if snapshot.get("eta"):
    fields.append(f"eta {format_duration(snapshot['eta'])}")
  if rate := snapshot.get("rate"):
    fields.append(f"{rate:.1f}/s")
  if progress["in_flight"]:
    fields.append(f"{progress['in_flight']} in flight")
  if (occupancy := snapshot.get("occupancy", {}).get("final")) is not None:
    fields.append(f"{occupancy * 100:.1f}% flagged")

  # Reserve the bar before laying out the fields, at a width that depends only
  # on the terminal. Sizing it from the leftover space instead would make it
  # grow and shrink as fields come and go.
  cells = min(MAX_BAR_CELLS, max(MIN_BAR_CELLS, width // 3))
  line = f"{head}[{render_bar(fraction, cells, glyphs)}] {percent}"

  if len(line) > width:
    line = f"{head}{percent}"

  for extra in (f"  {field}" for field in fields):
    if len(line) + len(extra) > width:
      break

    line += extra

  return line[:width]


class ProgressPrinter:
  """Writes a progress line, adapting to whether anything is watching.

  On a terminal the line is rewritten in place several times a second. When
  the output is redirected -- a stimela cab's captured stdout, a CI log, a
  nohup file -- carriage returns produce an unreadable single line, so the
  same text is emitted periodically as ordinary lines instead, and only when
  it has something new to say.
  """

  def __init__(
    self,
    stream: TextIO,
    interactive: bool,
    interval: float,
    glyphs: BarGlyphs = UNICODE_GLYPHS,
    width: int | None = None,
  ):
    self._stream = stream
    self._interactive = interactive
    self._glyphs = glyphs
    self._width = width
    self.interval = interval
    self._painted = 0
    self._last_line: str | None = None
    self._closed = False

  @classmethod
  def for_stream(cls, stream: TextIO, interval: float | None = None) -> ProgressPrinter:
    interactive = bool(getattr(stream, "isatty", lambda: False)())
    default = INTERACTIVE_INTERVAL_S if interactive else LOG_INTERVAL_S

    return cls(
      stream=stream,
      interactive=interactive,
      interval=default if interval is None else interval,
      glyphs=glyphs_for(stream),
    )

  def _terminal_width(self) -> int:
    if self._width is not None:
      return self._width

    # Re-read on every paint so that resizing a window does not leave a
    # trail of half-erased bars behind.
    return shutil.get_terminal_size((DEFAULT_WIDTH, 24)).columns

  def update(self, snapshot: Dict[str, Any]) -> None:
    if self._closed:
      return

    line = format_progress(snapshot, self._terminal_width() - 1, self._glyphs)

    if self._interactive:
      # Pad to cover whatever the previous, possibly longer, line left behind.
      self._stream.write("\r" + line.ljust(self._painted))
      self._painted = len(line)
    elif line == self._last_line:
      return
    else:
      self._stream.write(line + "\n")

    self._last_line = line
    self._stream.flush()

  def close(self, snapshot: Dict[str, Any] | None = None) -> None:
    """Paint a final line and leave the cursor on a fresh row."""
    if self._closed:
      return

    if snapshot is not None:
      self._last_line = None
      self.update(snapshot)

    if self._interactive and self._painted:
      self._stream.write("\n")
      self._stream.flush()

    self._closed = True
