"""
math_tool.py
------------
run_math_tool(expression) -> {"success": bool, "result_str": str | None, ...}

The LLM already passes a clean arithmetic expression (see the `calculate`
tool in src/agent/tools.py), so this evaluates the WHOLE string with a
small AST-based evaluator instead of regex-extracting a fragment of it.
The old regex dropped parentheses and thousands separators, e.g.
"(100+50)*2" became "100+50" and "1,234.50 + 10" became "234.50 + 10".

Uses Decimal so money figures don't pick up float noise
(-36.79 + 37.20 -> 0.41, not 0.41000000000000014). Never raises.
"""
from __future__ import annotations

import ast
import operator
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict

_BINARY_OPS: Dict[type, Callable[[Decimal, Decimal], Decimal]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: Dict[str, Callable[..., Decimal]] = {
    "sqrt": lambda x: x.sqrt(),
    "abs": abs,
    "round": lambda x, n=Decimal(0): round(x, int(n)),
}

_MAX_EXPONENT = 1000  # stops "9**9**9"-style hangs
_MAX_EXPR_LEN = 500
_DISPLAY_PLACES = Decimal("1e-10")


def run_math_tool(expression: str) -> Dict[str, Any]:
    expr_str = _clean(expression)
    if not expr_str:
        return {"success": False, "error": "Empty expression.", "result_str": None}
    if len(expr_str) > _MAX_EXPR_LEN:
        return {"success": False, "error": "Expression is too long.", "result_str": None}

    try:
        tree = ast.parse(expr_str, mode="eval")
        result = _eval(tree.body)
    except ZeroDivisionError:
        return {"success": False, "error": "Division by zero.", "result_str": None}
    except (SyntaxError, ValueError, TypeError, InvalidOperation, OverflowError) as exc:
        return {"success": False, "error": f"Could not evaluate: {exc}", "result_str": None}

    return {
        "success": True,
        "expression": expr_str,
        "result": float(result),
        "result_str": _format(result),
    }


def _clean(expression: str) -> str:
    s = (expression or "").strip()
    s = re.sub(r"[£$€₹]", "", s)                 # currency symbols
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)  # 1,234.50 -> 1234.50
    s = s.replace("^", "**").replace("×", "*").replace("÷", "/")
    return s


def _eval(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        # str() first so 0.1 stays Decimal("0.1"), not the float's long expansion
        return Decimal(str(node.value))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _BINARY_OPS[type(node.op)](left, right)

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCTIONS and not node.keywords:
        return _FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])

    raise ValueError("unsupported expression")


def _format(value: Decimal) -> str:
    text = format(value.quantize(_DISPLAY_PLACES), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text