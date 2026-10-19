"""Evaluate ASE unit expressions from config files (e.g. 'kcal / mol') without eval()."""

import ast
import math
import numbers
import operator

from ase import units

_OPS = {ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_NAMES = {k: float(v) for k, v in vars(units).items()
          if not k.startswith('_') and isinstance(v, numbers.Real) and not isinstance(v, bool)}
_NAMES['e'] = 1.0  # elementary charge, the charge unit SO3LR uses (not in ase.units)


def _eval(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)  # float, so huge powers overflow instead of hanging
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f'unsupported syntax: {ast.unparse(node)!r}')


def parse_unit_expression(value, field_name: str) -> float:
    """Return the conversion factor for ``value``, e.g. 'e * Angstrom' or 'kcal / mol'."""
    try:
        result = _eval(ast.parse(str(value).strip(), mode='eval').body)
        if not (math.isfinite(result) and result > 0):
            raise ValueError(f'must be finite and positive, got {result}')
    except (SyntaxError, TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(
            f"Could not parse unit string {value!r} for {field_name} ({exc}). Use ASE unit "
            f"names, numbers, *, / and **, e.g. 'eV', 'kcal / mol', 'e * Angstrom'.") from exc
    return result
