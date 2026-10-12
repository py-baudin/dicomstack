""" class for making queries

# usage
stack(Selector("SeriesNumber") == 301)
stack(Selector("ImageType")[0] == "M")
stack(DICOM.SeriesNumber.is_in([301, 401]))

# shortcuts
stack(stack.SeriesNumber == 301)
stack(stack.ImageType[0] == "M")
stack(stack.Field.Subfield == "value")

# operators:
==, !=, <, <=, >, >=

# methods:
.is_in() (alias: .isin()), .not_in(), .between(lo, hi) (inclusive),
.isclose(value, tol=1e-6), .exists(), .is_null(), .contains(), .intersect(),
.subset(), .superset(), .startswith(), .endswith(),
.regex(pattern) (re.search: matches anywhere; use "^" to anchor), .apply(function)

# combine/modify queries with &, |, ~

# errors
A comparison raising TypeError or AttributeError (e.g. `DICOM.SeriesNumber > "1"`, or a
missing value compared with `<`) excludes the frame with a warning;
`stack(query, strict=True)` raises instead.
"""

import re
import math
from typing import Any
import warnings


class Selector:
    """Store a keys for creating queries"""

    def __init__(self, *keys):
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        return f"<{'.'.join([str(k) for k in self.keys])}>"

    def __getitem__(self, index: int) -> "Selector":
        return Selector(*(self.keys + (index,)))

    def __getattr__(self, name: str) -> "Selector":
        if name.startswith("__"):
            # special attributes (copy, pickle, numpy...) are not DICOM fields
            raise AttributeError(name)
        return Selector(*(self.keys + (name,)))

    def __iter__(self):
        raise NotImplementedError()

    def __eq__(self, value: Any) -> "Query":
        return Query.from_selector(self, "equal", value)

    def __contains__(self, value: Any) -> "Query":
        return Query.from_selector(self, "contains", value)

    def __ne__(self, value: Any) -> "Query":
        return Query.from_selector(self, "not-equal", value)

    def __lt__(self, value: Any) -> "Query":
        return Query.from_selector(self, "less-than", value)

    def __le__(self, value: Any) -> "Query":
        return Query.from_selector(self, "less-or-equal", value)

    def __gt__(self, value: Any) -> "Query":
        return Query.from_selector(self, "greater-than", value)

    def __ge__(self, value: Any) -> "Query":
        return Query.from_selector(self, "greater-or-equal", value)

    # methods
    def is_in(self, value: Any) -> "Query":
        return Query.from_selector(self, "is-in", value)

    isin = is_in

    def not_in(self, value: Any) -> "Query":
        return Query.from_selector(self, "not-in", value)

    def between(self, low: Any, high: Any) -> "Query":
        """low <= value <= high"""
        return Query.from_selector(self, "between", (low, high))

    def isclose(self, value: Any, tol: float = 1e-6) -> "Query":
        """|value - reference| <= tol (absolute tolerance), element-wise for tuples"""
        return Query.from_selector(self, "is-close", (value, tol))

    def exists(self) -> "Query":
        """the field exists and is not empty (same as ~is_null())"""
        return Query.from_selector(self, "exists")

    def contains(self, value: Any) -> "Query":
        return Query.from_selector(self, "contains", value)

    def intersect(self, value: Any) -> "Query":
        return Query.from_selector(self, "intersect", value)

    def subset(self, value: Any) -> "Query":
        return Query.from_selector(self, "subset", value)

    def superset(self, value: Any) -> "Query":
        return Query.from_selector(self, "superset", value)

    def is_null(self) -> "Query":
        return Query.from_selector(self, "is-null")

    def startswith(self, value: Any) -> "Query":
        return Query.from_selector(self, "starts-with", value)

    def endswith(self, value: Any) -> "Query":
        return Query.from_selector(self, "ends-with", value)

    def regex(self, value: Any) -> "Query":
        """re.search: the pattern can match anywhere in the value (use "^" to anchor it)"""
        return Query.from_selector(self, "regex", value)

    def apply(self, expr: Any) -> "Query":
        return Query.from_selector(self, "apply", expr)


def is_close(value, reference, tol):
    if isinstance(reference, (tuple, list)):
        return len(value) == len(reference) and all(
            is_close(v, r, tol) for v, r in zip(value, reference)
        )
    return math.fabs(value - reference) <= tol


# comparison operators: (field's value, query value) -> result
OPERATORS = {
    "is-null": lambda value, _: value is None,
    "exists": lambda value, _: value is not None,
    "equal": lambda value, ref: value == ref,
    "not-equal": lambda value, ref: value != ref,
    "less-than": lambda value, ref: value < ref,
    "greater-than": lambda value, ref: value > ref,
    "less-or-equal": lambda value, ref: value <= ref,
    "greater-or-equal": lambda value, ref: value >= ref,
    "between": lambda value, ref: ref[0] <= value <= ref[1],
    "is-close": lambda value, ref: is_close(value, ref[0], ref[1]),
    "is-in": lambda value, ref: value in ref,
    "not-in": lambda value, ref: not value in ref,
    "contains": lambda value, ref: ref in value,
    "intersect": lambda value, ref: bool(set(value) & set(ref)),
    "subset": lambda value, ref: bool(set(value) <= set(ref)),
    "superset": lambda value, ref: bool(set(value) >= set(ref)),
    "starts-with": lambda value, ref: value.startswith(ref),
    "ends-with": lambda value, ref: value.endswith(ref),
    "regex": lambda value, ref: re.search(ref, value) is not None,
    "apply": lambda value, ref: ref(value),
}


class Query:
    """Store a query or a combination of queries

    A tree of nodes: ("const", bool), ("compare", selector, op, value),
    ("and", query, query), ("or", query, query), ("not", query).
    Query(True) passes all frames, Query(False) none.
    """

    def __init__(self, expr=True):
        self.node = ("const", bool(expr))

    @classmethod
    def _make(cls, *node):
        query = cls.__new__(cls)
        query.node = node
        return query

    @classmethod
    def from_selector(cls, selector, op, value=None) -> "Query":
        if not op in OPERATORS:
            raise ValueError(f"Unknown operator: {op}")
        return cls._make("compare", selector, op, value)

    @property
    def constant(self):
        """True or False for constant queries (All, None), else None"""
        return self.node[1] if self.node[0] == "const" else None

    def __repr__(self):
        kind = self.node[0]
        if kind == "const":
            return "All" if self.node[1] else "None"
        elif kind == "compare":
            _, selector, op, value = self.node
            return f"{'.'.join([str(k) for k in selector.keys])}<{op}>{value}"
        elif kind == "not":
            return f"not ({self.node[1]!r})"
        return f"({self.node[1]!r}) {kind} ({self.node[2]!r})"

    @staticmethod
    def compare(op: str, value1: Any, value2: Any) -> bool:
        """compare a field's value (value1) with a query value (value2)"""
        try:
            operator = OPERATORS[op]
        except KeyError:
            raise ValueError(f"Unknown operator: {op}")
        return operator(value1, value2)

    def execute(self, getter, strict=False):
        """execute query, return boolean value

        getter: function returning a field's value for a Selector (e.g. frame.get)
        strict: raise errors of the comparisons (else: warn and return False)
        """
        kind = self.node[0]
        if kind == "const":
            return self.node[1]
        elif kind == "compare":
            _, selector, op, value = self.node
            try:
                return OPERATORS[op](getter(selector), value)
            except (TypeError, AttributeError) as exc:
                if strict:
                    raise
                from .pixeldata import external_stacklevel  # (numpy import)

                warnings.warn(
                    f"`{selector}{op}({value})` returned error: {exc}",
                    stacklevel=external_stacklevel(),
                )
                return False
        elif kind == "and":
            return self.node[1].execute(getter, strict) and self.node[2].execute(getter, strict)
        elif kind == "or":
            return self.node[1].execute(getter, strict) or self.node[2].execute(getter, strict)
        elif kind == "not":
            return not self.node[1].execute(getter, strict)
        raise ValueError(f"Invalid query node: {kind}")

    def __call__(self, *args, **kwargs):
        """short for execute"""
        return self.execute(*args, **kwargs)

    def __and__(self, other: "Query") -> "Query":
        if self.constant is True:
            return other
        elif other.constant is True:
            return self
        elif self.constant is False or other.constant is False:
            return Query(False)
        return Query._make("and", self, other)

    def __or__(self, other: "Query") -> "Query":
        if self.constant is False:
            return other
        elif other.constant is False:
            return self
        elif self.constant is True or other.constant is True:
            return Query(True)
        return Query._make("or", self, other)

    def __invert__(self) -> "Query":
        if self.constant is not None:
            return Query(not self.constant)
        return Query._make("not", self)
