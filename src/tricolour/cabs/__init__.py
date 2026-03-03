"""Cab definitions for generated Stimela cabs."""

from pathlib import Path

CAB_DIR = Path(__file__).parent
AVAILABLE_CABS = [p.stem for p in CAB_DIR.glob("*.yml")]


def get_cab_path(name: str) -> Path:
    """Get path to a cab definition.

    Args:
        name: Name of the cab (without .yml extension)

    Returns:
        Path to the cab YAML file

    Raises:
        FileNotFoundError: If the cab doesn't exist
    """
    cab_path = CAB_DIR / f"{name}.yml"
    if not cab_path.exists():
        raise FileNotFoundError(f"Cab not found: {name}")
    return cab_path


__all__ = ["CAB_DIR", "AVAILABLE_CABS", "get_cab_path"]
