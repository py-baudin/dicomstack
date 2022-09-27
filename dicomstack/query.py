""" class for making queries

# usage
stack.query(Selector("SeriesNumber").equals(301))
stack.query(Selector("ImageType")[0].equals("M"))

# shortcuts
stack(stack.SeriesNumber == 301)
stack(stack.ImageType[0] == "M")
stack(stack.Field.Subfield == "value")

stack("SeriesNumber == 301")
stack("SeriesNumber[0] == 'M'")

# operators:
==, !=, <, <=, >, >=

# methods:
.is_in(), .not_in(), is_null(), contains(), startswith(), endswith(), between(), regex(),

# combine/modify queries with &, |, ~

"""

import re
from typing import Any


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
        try:
            getattr(super(), name)
        except AttributeError:
            return Selector(*(self.keys + (name,)))

    def __iter__(self):
        raise NotImplementedError()

    def __eq__(self, value: Any) -> "Query":
        return Query.from_selector(self, "equal", value)

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

    def not_in(self, value: Any) -> "Query":
        return Query.from_selector(self, "not-in", value)

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
        return Query.from_selector(self, "regex", value)


class Query:
    """Store a query or a combination of queries"""

    def __init__(self, expr=True, data=None):
        self.expr = expr if isinstance(expr, str) else bool(expr)
        self.data = data if data else {}

    def __repr__(self):
        if self.expr is True:
            return "All"
        elif self.expr is False:
            return "None"
        expr = self.expr
        for id, data in self.data.items():
            expr = expr.replace(id + "()", data["repr"])
        return expr

    # @classmethod
    # def from_string(self, string) -> "Query":

    @classmethod
    def from_selector(cls, selector, op, value=None) -> "Query":
        id = f"__{abs(hash(str(selector.keys) + str(op) + str(value)))}__"
        repr = f"{'.'.join([str(k) for k in selector.keys])}<{op}>{value}"
        expr = f"{id}()"
        data = {id: {"selector": selector, "op": op, "value": value, "repr": repr}}
        return cls(expr, data)

    def compare(self, op: str, value1: Any, value2: Any) -> bool:
        if op == "is-null":
            if value1 is None:
                return True
            return False

        elif op == "equal":
            return value1 == value2
        elif op == "not-equal":
            return value1 != value2
        elif op == "less-than":
            return value1 < value2
        elif op == "greater-than":
            return value1 > value2
        elif op == "less-or-equal":
            return value1 <= value2
        elif op == "greater-or-equal":
            return value1 >= value2
        elif op == "is-in":
            return value1 in value2
        elif op == "not-in":
            return not value1 in value2
        elif op == "contains":
            return value2 in value1
        elif op == "intersect":
            return bool(set(value1) & set(value2))
        elif op == "subset":
            return bool(set(value1) <= set(value2))
        elif op == "superset":
            return bool(set(value1) >= set(value2))
        elif op == "starts-with":
            return value1.startswith(value2)
        elif op == "ends-with":
            return value1.endswith(value2)
        elif op == "regex":
            return re.match(value2, value1) is not None
        else:
            raise ValueError(f"Unknown operator: {op}")

    def execute(self, getter):
        """execute query, return boolean value"""
        if self.expr is True: # pass all
            return True
        elif self.expr is False: # pass none
            return False

        def get_callback(getter, data):
            def callback():
                return self.compare(data["op"], getter(data["selector"]), data["value"])

            return callback

        callbacks = {id: get_callback(getter, data) for id, data in self.data.items()}
        return eval(self.expr, {}, callbacks)

    def __call__(self, *args):
        """short for execute"""
        return self.execute(*args)

    def __and__(self, other: "Query") -> "Query":
        if self.expr is True:
            return other
        elif other.expr is True:
            return self
        elif self.expr is False or other.expr is False:
            return Query(False)
        expr = f"({self.expr}) and ({other.expr})"
        data = {**self.data, **other.data}
        return Query(expr, data)

    def __or__(self, other: "Query") -> "Query":
        if self.expr is False:
            return other
        elif other.expr is False:
            return self
        elif self.expr is True or other.expr is True:
            return Query(True)
        expr = f"({self.expr}) or ({other.expr})"
        data = {**self.data, **other.data}
        return Query(expr, data)

    def __invert__(self) -> "Query":
        if isinstance(self.expr, bool):
            return Query(not self.expr)
        return Query(f"not ({self.expr})", self.data)
