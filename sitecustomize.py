import warnings
from holidays.deprecations.v1_incompatibility import FutureIncompatibilityWarning # type: ignore

warnings.simplefilter("ignore", FutureIncompatibilityWarning)
