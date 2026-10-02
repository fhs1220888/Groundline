"""groundline — a verifiable test-data analysis agent."""

__version__ = "0.1.3"

from .session import Evidence, Session  # noqa: E402

__all__ = ["Session", "Evidence", "__version__"]
