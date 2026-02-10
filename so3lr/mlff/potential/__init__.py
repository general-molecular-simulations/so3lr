"""
Backward compatibility module.

This module re-exports from mlff.calculators.potential for compatibility.
The implementation has been consolidated into mlff.calculators.potential
to eliminate code duplication.

For new code, consider importing directly from mlff.calculators.potential:
    from mlff.calculators.potential import PotentialSparse

The old API is maintained for backward compatibility:
    from mlff.potential import MLFFPotential
"""

from ..calculators.potential import (
    MachineLearningPotential,
    PotentialSparse as MLFFPotentialSparse,
)

# Backward compatibility alias
MLFFPotential = MLFFPotentialSparse

__all__ = ['MachineLearningPotential', 'MLFFPotentialSparse', 'MLFFPotential']
