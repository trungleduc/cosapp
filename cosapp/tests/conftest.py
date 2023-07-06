import pytest
import sys
from pathlib import Path


@pytest.fixture
def test_library():
    library_path = Path(__file__).parent / "library" / "systems"

    # Add path to allow System to find the component
    sys.path.append(str(library_path))
    try:
        yield library_path
    finally:
        # Undo path modification
        sys.path.remove(str(library_path))


@pytest.fixture
def test_data():
    return Path(__file__).parent / "data"
