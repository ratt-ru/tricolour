import pytest

from tricolour.core.util import normalize_chunks


@pytest.mark.parametrize(
  "chunks, shape, result",
  [
    # --- Test 1: Uniform Integer Chunks ---
    # 25 split by 10 -> (10, 10, 5)
    ([10], [25], ((10, 10, 5),)),
    # --- Test 2: Underfill (Expansion) ---
    # Shape 35, Chunks (10, 10). Last value is 10.
    # Needs 15 more -> adds 10, then 5.
    ([(10, 10)], [35], ((10, 10, 10, 5),)),
    # --- Test 3: Overfill (Truncation) ---
    # Shape 30, Chunks (20, 20).
    # Second 20 is truncated to 10.
    ([(20, 20)], [30], ((20, 10),)),
    # --- Test 4: Multi-dimensional Mixed ---
    # Dim 0: Uniform 50 for 100
    # Dim 1: Explicit (30,) for 70 -> Last val 30 fills to (30, 30, 10)
    ((50, (30,)), (100, 70), ((50, 50), (30, 30, 10))),
    # --- Test 5: Edge Cases ---
    # Zero shape
    ((10, (5,)), (0, 0), ((), ())),
    # Empty iterable for positive shape -> returns shape as one chunk
    ([[]], [15], ((15,),)),
  ],
)
def test_normalize_chunks(chunks, shape, result):
  assert normalize_chunks(chunks, shape) == result
