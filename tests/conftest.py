from __future__ import annotations

import pandas as pd
import pytest

from helpers import make_synthetic


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()
