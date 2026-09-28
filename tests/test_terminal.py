"""Tests for the terminal renderer over a run snapshot."""

import io

import pytest

from tricolour.core.application.terminal import (
  ASCII_GLYPHS,
  MIN_BAR_CELLS,
  UNICODE_GLYPHS,
  ProgressPrinter,
  describe_status,
  format_duration,
  format_progress,
  glyphs_for,
  render_bar,
)

pytestmark = pytest.mark.unit


def snapshot(**overrides):
  """A snapshot shaped like the one RunState.snapshot() produces."""
  progress = {
    "total_items": 100,
    "dispatched": 40,
    "completed": 25,
    "in_flight": 15,
    "total_visibilities": 1600,
    "completed_visibilities": 400,
    "fraction": 0.25,
  }
  progress.update(overrides.pop("progress", {}))

  state = {
    "running": True,
    "finished_at": None,
    "elapsed": 30.0,
    "eta": 90.0,
    "rate": 0.83,
    "progress": progress,
    "occupancy": {"original": 0.4, "final": 0.52},
    "errors": [],
  }
  state.update(overrides)
  return state


@pytest.mark.parametrize(
  "seconds, expected",
  [(None, "--"), (0, "0s"), (9.4, "9s"), (59.6, "1m00s"), (95, "1m35s"), (3600, "1h00m"), (7325, "2h02m")],
)
def test_format_duration(seconds, expected):
  assert format_duration(seconds) == expected


def test_format_duration_handles_infinity():
  assert format_duration(float("inf")) == "--"


@pytest.mark.parametrize("cells", [1, 8, 20, 73])
@pytest.mark.parametrize("fraction", [0.0, 0.01, 0.25, 0.5, 0.99, 1.0])
def test_render_bar_is_always_exactly_the_requested_width(cells, fraction):
  """A bar that is not exactly `cells` wide leaves debris on the terminal."""
  assert len(render_bar(fraction, cells, UNICODE_GLYPHS)) == cells
  assert len(render_bar(fraction, cells, ASCII_GLYPHS)) == cells


def test_render_bar_endpoints():
  assert render_bar(0.0, 4, ASCII_GLYPHS) == "----"
  assert render_bar(1.0, 4, ASCII_GLYPHS) == "####"
  assert render_bar(0.5, 4, ASCII_GLYPHS) == "##--"


def test_render_bar_clamps_out_of_range_fractions():
  assert render_bar(-0.5, 4, ASCII_GLYPHS) == "----"
  assert render_bar(1.5, 4, ASCII_GLYPHS) == "####"


def test_render_bar_uses_partial_blocks_for_sub_cell_progress():
  """A long run must visibly advance between whole cells."""
  coarse = render_bar(0.26, 10, ASCII_GLYPHS)
  fine = render_bar(0.26, 10, UNICODE_GLYPHS)

  assert coarse == render_bar(0.20, 10, ASCII_GLYPHS)  # ASCII cannot tell them apart
  assert fine != render_bar(0.20, 10, UNICODE_GLYPHS)
  assert len(fine) == 10


def test_render_bar_of_zero_cells_is_empty():
  assert render_bar(0.5, 0, UNICODE_GLYPHS) == ""


@pytest.mark.parametrize(
  "overrides, expected",
  [
    ({}, "flagging"),
    ({"errors": ["RuntimeError: boom"]}, "failed"),
    ({"finished_at": 123.0}, "done"),
    ({"running": False}, "waiting"),
    ({"progress": {"total_items": 0}}, "planning"),
    # An error outranks completion: a finished run with failures is not "done"
    ({"finished_at": 123.0, "errors": ["boom"]}, "failed"),
  ],
)
def test_describe_status(overrides, expected):
  assert describe_status(snapshot(**overrides)) == expected


def test_format_progress_reports_the_essentials():
  line = format_progress(snapshot(), width=100, glyphs=ASCII_GLYPHS)

  assert "flagging" in line
  assert " 25%" in line
  assert "25/100" in line
  assert "eta 1m30s" in line
  assert "0.8/s" in line
  assert "15 in flight" in line
  assert "52.0% flagged" in line


@pytest.mark.parametrize("width", range(12, 121, 4))
def test_format_progress_never_exceeds_the_terminal_width(width):
  """Overrunning the width wraps the line and leaves a trail of stale bars."""
  for overrides in ({}, {"progress": {"total_items": 0}}, {"finished_at": 1.0}, {"errors": ["x"]}):
    line = format_progress(snapshot(**overrides), width=width, glyphs=ASCII_GLYPHS)
    assert len(line) <= width, (width, line)


def test_format_progress_drops_fields_before_it_drops_the_bar():
  wide = format_progress(snapshot(), width=100, glyphs=ASCII_GLYPHS)
  narrow = format_progress(snapshot(), width=44, glyphs=ASCII_GLYPHS)

  assert "52.0% flagged" in wide
  assert "52.0% flagged" not in narrow
  assert "[" in narrow and "25%" in narrow


def test_bar_width_does_not_change_as_fields_come_and_go():
  """A bar sized from the leftover space visibly jumps as `eta` appears.

  The line is repainted several times a second, so anything that resizes
  between paints reads as the bar bouncing around.
  """

  def bar_width(state):
    line = format_progress(state, width=80, glyphs=ASCII_GLYPHS)
    return len(line[line.index("[") + 1 : line.index("]")])

  variants = [
    snapshot(),
    snapshot(eta=None),
    snapshot(rate=None, progress={"in_flight": 0}),
    snapshot(occupancy={"original": None, "final": None}),
    snapshot(finished_at=1.0, running=False, progress={"completed": 100, "fraction": 1.0, "in_flight": 0}),
  ]

  assert len({bar_width(state) for state in variants}) == 1


def test_format_progress_drops_the_bar_when_there_is_no_room_left():
  line = format_progress(snapshot(), width=14, glyphs=ASCII_GLYPHS)

  assert "[" not in line
  assert "25%" in line


def test_format_progress_before_the_work_is_counted():
  """`plan` runs after the clock starts, so there is a window with no total."""
  line = format_progress(snapshot(progress={"total_items": 0}), width=60, glyphs=ASCII_GLYPHS)

  assert line == "planning 30s"


def test_format_progress_omits_fields_that_are_absent():
  line = format_progress(
    snapshot(eta=None, rate=None, occupancy={"original": None, "final": None}, progress={"in_flight": 0}),
    width=100,
    glyphs=ASCII_GLYPHS,
  )

  assert "eta" not in line
  assert "/s" not in line
  assert "in flight" not in line
  assert "flagged" not in line
  assert "25/100" in line


def test_format_progress_of_a_finished_run_shows_a_full_bar():
  line = format_progress(
    snapshot(running=False, finished_at=1.0, eta=0.0, progress={"completed": 100, "fraction": 1.0, "in_flight": 0}),
    width=100,
    glyphs=ASCII_GLYPHS,
  )

  assert "done" in line
  assert "100%" in line
  assert ASCII_GLYPHS.empty * 2 not in line


class FakeStream:
  """A writable stream with a settable encoding, which StringIO is not."""

  def __init__(self, tty, encoding="utf-8"):
    self._buffer = io.StringIO()
    self._tty = tty
    self.encoding = encoding

  def write(self, text):
    return self._buffer.write(text)

  def flush(self):
    pass

  def isatty(self):
    return self._tty

  def getvalue(self):
    return self._buffer.getvalue()

  def reset(self):
    self._buffer.seek(0)
    self._buffer.truncate(0)


def test_glyphs_fall_back_when_the_stream_cannot_encode_them():
  """A C-locale container must not take the run down over a progress bar."""
  assert glyphs_for(FakeStream(True, encoding="ascii")) is ASCII_GLYPHS
  assert glyphs_for(FakeStream(True, encoding="utf-8")) is UNICODE_GLYPHS
  assert glyphs_for(FakeStream(True, encoding=None)) is ASCII_GLYPHS


def test_interactive_output_rewrites_one_line():
  stream = FakeStream(tty=True)
  printer = ProgressPrinter(stream, interactive=True, interval=0.1, glyphs=ASCII_GLYPHS, width=60)

  printer.update(snapshot())
  printer.update(snapshot(progress={"completed": 50, "fraction": 0.5}))

  written = stream.getvalue()
  assert written.count("\r") == 2
  assert "\n" not in written

  printer.close()
  assert stream.getvalue().endswith("\n")


def test_interactive_output_erases_a_longer_previous_line():
  """Shrinking content must not leave the tail of the last line behind."""
  stream = FakeStream(tty=True)
  printer = ProgressPrinter(stream, interactive=True, interval=0.1, glyphs=ASCII_GLYPHS, width=60)

  printer.update(snapshot())
  long_line = len(stream.getvalue()) - 1
  stream.reset()

  printer.update(snapshot(progress={"total_items": 0}))
  assert len(stream.getvalue().lstrip("\r")) == long_line


def test_redirected_output_emits_whole_lines():
  """Carriage returns in a captured log produce one unreadable line."""
  stream = FakeStream(tty=False)
  printer = ProgressPrinter(stream, interactive=False, interval=30.0, glyphs=ASCII_GLYPHS, width=60)

  printer.update(snapshot())
  printer.update(snapshot(progress={"completed": 50, "fraction": 0.5}))

  written = stream.getvalue()
  assert "\r" not in written
  assert len(written.splitlines()) == 2


def test_redirected_output_does_not_repeat_itself():
  """A stalled run should not fill a log file with identical lines."""
  stream = FakeStream(tty=False)
  printer = ProgressPrinter(stream, interactive=False, interval=30.0, glyphs=ASCII_GLYPHS, width=60)

  for _ in range(5):
    printer.update(snapshot())

  assert len(stream.getvalue().splitlines()) == 1


def test_close_paints_a_final_snapshot_even_if_unchanged():
  stream = FakeStream(tty=False)
  printer = ProgressPrinter(stream, interactive=False, interval=30.0, glyphs=ASCII_GLYPHS, width=60)

  printer.update(snapshot())
  printer.close(snapshot(finished_at=1.0, running=False, progress={"completed": 100, "fraction": 1.0}))

  lines = stream.getvalue().splitlines()
  assert len(lines) == 2
  assert "done" in lines[-1]


def test_close_is_idempotent():
  stream = FakeStream(tty=True)
  printer = ProgressPrinter(stream, interactive=True, interval=0.1, glyphs=ASCII_GLYPHS, width=60)

  printer.update(snapshot())
  printer.close()
  written = stream.getvalue()

  printer.close(snapshot())
  printer.update(snapshot())
  assert stream.getvalue() == written


def test_for_stream_picks_a_slow_cadence_when_redirected():
  assert ProgressPrinter.for_stream(FakeStream(tty=True)).interval < 1.0
  assert ProgressPrinter.for_stream(FakeStream(tty=False)).interval >= 10.0


def test_min_bar_cells_is_respected():
  line = format_progress(snapshot(), width=30, glyphs=ASCII_GLYPHS)

  if "[" in line:
    assert len(line[line.index("[") + 1 : line.index("]")]) >= MIN_BAR_CELLS
