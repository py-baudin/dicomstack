""" query engine (phase 4): operators, combinations, errors, strict mode """

import copy
import pickle
import warnings

import pytest

from dicomstack import DicomStack, DICOM
from dicomstack.query import Query, Selector

ROWS = [
    {"a": 1, "b": (1, 2), "c": "foobar", "e": 1.0000001, "p": "IMAGE"},
    {"a": 2, "b": (1, 3), "c": "bazfoo", "e": 2.5, "d": "x"},
    {"a": 3, "b": (4,), "c": "FOO", "e": -1.0, "d": None},
]


def getter(row):
    """as frame.get: None for missing fields"""

    def get(selector):
        value = row
        for key in selector.keys:
            try:
                value = value[key]
            except (KeyError, IndexError):
                return None
        return value

    return get


def run(query, row):
    """result and number of warnings"""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = query.execute(getter(row))
    return result, len(record)


def test_operators():
    a, b, c, d = Selector("a"), Selector("b"), Selector("c"), Selector("d")
    expected = [
        (a == 1, [True, False, False]),
        (a != 1, [False, True, True]),
        (a < 2, [True, False, False]),
        (a >= 2, [False, True, True]),
        (a.is_in([1, 3]), [True, False, True]),
        (a.not_in([1, 3]), [False, True, False]),
        (b[1] == 3, [False, True, False]),
        (b.contains(2), [True, False, False]),
        (b.intersect([2, 4]), [True, False, True]),
        (b.subset([1, 2, 5]), [True, False, False]),
        (b.superset([1]), [True, True, False]),
        (c.startswith("foo"), [True, False, False]),
        (c.endswith("foo"), [False, True, False]),
        (d.is_null(), [True, False, True]),
        (d == "x", [False, True, False]),
        (a.apply(lambda value: value % 2 == 1), [True, False, True]),
        (Selector("p")[0] == "I", [True, False, False]),
    ]
    for query, results in expected:
        assert [query.execute(getter(row)) for row in ROWS] == results, repr(query)


def test_combinations():
    a, c, d = Selector("a"), Selector("c"), Selector("d")
    row = ROWS[0]
    # errors (None < 1): warning, False; and/or short-circuit
    assert run(d < 1, row) == (False, 1)
    assert run((a < 3) & (d < 1), row) == (False, 1)
    assert run((a > 5) & (d < 1), row) == (False, 0)
    assert run((d < 1) | (a < 3), row) == (True, 1)
    assert run((a < 3) | (d < 1), row) == (True, 0)
    assert run(~((a == 1) & c.regex("^foo")), row) == (False, 0)

    query = ~((a == 1) & c.regex("^foo")) | (d < 1)
    assert repr(query) == "(not ((a<equal>1) and (c<regex>^foo))) or (d<less-than>1)"

    # constants
    q = a == 1
    assert repr(Query(True)) == "All" and repr(~Query(True)) == "None"
    assert (Query(True) & q) is q and (q | Query(False)) is q
    assert repr(q & Query(False)) == "None" and repr(q | Query(True)) == "All"


def test_no_collisions():
    # the old engine identified conditions by hash(str(...)): 1 and "1" collided
    query = (Selector("a") == 1) | (Selector("a") == "1")
    assert [query.execute(getter({"a": value})) for value in [1, "1", 2]] == [True, True, False]


def test_new_methods():
    a, b, c, e = Selector("a"), Selector("b"), Selector("c"), Selector("e")
    results = lambda query: [bool(query.execute(getter(row))) for row in ROWS]
    assert results(a.between(2, 3)) == [False, True, True]
    assert results(e.isclose(1.0, tol=1e-6)) == [True, False, False]
    assert results(e.isclose(1.0, tol=1e-9)) == [False, False, False]
    assert results(b.isclose((1, 2.0000001))) == [True, False, False]
    assert results(Selector("d").exists()) == [False, True, False]
    # regex: re.search (anywhere in the value), "^" to anchor
    assert results(c.regex("foo")) == [True, True, False]
    assert results(c.regex("^foo")) == [True, False, False]
    assert not hasattr(Selector, "search")
    assert results(a.isin([2])) == [False, True, False]
    assert repr(a.between(2, 3)) == "a<between>(2, 3)"


def test_strict():
    query = Selector("c") > 1
    with pytest.warns(UserWarning, match="returned error"):
        assert query.execute(getter(ROWS[0])) is False
    with pytest.raises(TypeError):
        query.execute(getter(ROWS[0]), strict=True)


def test_selector_dunders():
    sel = Selector("a")[0].b
    with pytest.raises(AttributeError):
        sel.__deepcopy__
    assert copy.copy(sel).keys == sel.keys
    assert copy.deepcopy(sel).keys == sel.keys
    assert pickle.loads(pickle.dumps(sel)).keys == ("a", 0, "b")
    assert not hasattr(sel, "__array__")
    # attributes of object: still selectors
    assert sel.keys == ("a", 0, "b")
    assert Selector().format.keys == ("format",)


def test_query_pickle():
    query = ((Selector("a") == 1) & ~Selector("c").regex("x")) | Query(False)
    query2 = pickle.loads(pickle.dumps(query))
    assert repr(query2) == repr(query)
    assert [query2.execute(getter(row)) for row in ROWS] == [query.execute(getter(row)) for row in ROWS]


#
# on a stack


def test_stack_queries(legsfile, multi):
    stack = DicomStack(legsfile)
    assert stack(DICOM["[CSAImageHeaderType]"] == "IMAGE NUM 4")
    assert stack(DICOM["[TablePositionOrigin]"][2].between(-1500, -1400))
    assert not stack(DICOM["[GradientMode]"].regex("^Fast"))
    assert stack(DICOM.PixelSpacing.isclose((1.484375, 1.484375)))
    assert stack(DICOM.Modality.regex("R"))  # anywhere
    assert not stack(DICOM.Modality.regex("^R"))
    assert stack(DICOM.Modality.exists())
    assert not stack(DICOM.StudyID.exists())  # empty value
    assert not stack(DICOM.UnknownField.exists())
    assert stack(stack.SeriesNumber == 3)
    assert stack.select((0x0020, 0x0011)).keys == DICOM[(0x0020, 0x0011)].keys

    multi = DicomStack(multi)
    echo_times = multi.unique("EchoTime")
    assert len(multi(DICOM.EchoTime.between(echo_times[1], echo_times[3]))) == 15
    assert len(multi(DICOM.EchoTime.isclose(echo_times[0], tol=1e-3))) == 5


def test_filter_missing_index(legsfile):
    # out-of-range index: frame skipped, as a missing field (and as frame.get)
    stack = DicomStack(legsfile)
    assert stack.frames[0].get("ImageType_9") is None
    assert len(stack(ImageType_9="X")) == 0
    assert len(stack(UnknownField="X")) == 0
    assert len(stack(ImageType_1="ORIGINAL")) == 1


def test_stack_strict(legsfile):
    stack = DicomStack(legsfile)
    with pytest.warns(UserWarning, match="returned error") as record:
        assert not stack(DICOM.SeriesNumber > "1")
    assert record[0].filename == __file__
    with pytest.raises(TypeError):
        stack(DICOM.SeriesNumber > "1", strict=True)
    # missing values
    with pytest.warns(UserWarning):
        assert not stack(DICOM.StudyID < 1)
    with pytest.raises(TypeError):
        stack(DICOM.StudyID < 1, strict=True)
