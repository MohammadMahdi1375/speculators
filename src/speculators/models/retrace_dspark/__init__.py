from speculators.config import reload_schemas

from .config import ReTraceDSparkSpeculatorConfig
from .core import ReTraceDSparkDraftModel

# This package can be imported after speculators' initial registry rebuild.
# Rebuild inherited Transformers/Pydantic annotations and include this identity
# without editing global registration files or the running DFlash package.
reload_schemas()

__all__ = ["ReTraceDSparkDraftModel", "ReTraceDSparkSpeculatorConfig"]
