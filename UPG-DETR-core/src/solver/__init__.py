from typing import Dict

from ._solver import BaseSolver
from .det_solver import DetSolver

TASKS: Dict[str, BaseSolver] = {"detection": DetSolver}
