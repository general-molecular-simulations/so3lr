"""Safe parsing for unit expressions used in training configuration files."""

from __future__ import annotations

import ast
import math
import numbers
import operator

from ase import units as ase_units


_BINARY_OPERATORS = {
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# ASE exposes a few non-unit helpers alongside its numerical constants. Only
# immutable real-valued constants are made visible to the expression evaluator.
_UNIT_NAMES = {
    name: float(value)
    for name in dir(ase_units)
    if not name.startswith("_")
    and isinstance((value := getattr(ase_units, name)), numbers.Real)
    and not isinstance(value, bool)
}
_UNIT_NAMES["e"] = 1.0  # charge labels are expressed in elementary-charge units


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, numbers.Real):
            raise ValueError("only real numeric constants are allowed")
        return float(node.value)

    if isinstance(node, ast.Name):
        try:
            return _UNIT_NAMES[node.id]
        except KeyError as exc:
            raise ValueError(f"unknown unit name {node.id!r}") from exc

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (not float(right).is_integer() or abs(right) > 12):
            raise ValueError("unit exponents must be integers between -12 and 12")
        return _BINARY_OPERATORS[type(node.op)](left, right)

    raise ValueError(
        "only unit names, real numbers, parentheses, multiplication, division, "
        "and small integer powers are allowed"
    )


def parse_unit_expression(value, *, field_name: str) -> float:
    """Return a finite, positive conversion factor without executing code.

    Expressions may contain numerical constants and real-valued names exported
    by :mod:`ase.units`, plus ``e`` for the elementary-charge convention used by
    SO3LR. Function calls, attribute access, indexing, and other Python syntax
    are deliberately rejected.
    """
    if isinstance(value, bool) or not isinstance(value, (str, numbers.Real)):
        raise ValueError(f"{field_name} must be a unit expression or real number")

    expression = str(value).strip()
    if not expression or len(expression) > 256:
        raise ValueError(f"{field_name} must be a non-empty unit expression of at most 256 characters")

    try:
        parsed = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse unit string {expression!r} for {field_name}") from exc

    if sum(1 for _ in ast.walk(parsed)) > 64:
        raise ValueError(f"Unit expression for {field_name} is too complex")

    try:
        result = float(_evaluate(parsed))
    except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse unit string {expression!r} for {field_name}: {exc}") from exc

    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Unit expression for {field_name} must evaluate to a finite positive number")
    return result
