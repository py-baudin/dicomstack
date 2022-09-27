""" unittest for dicomstack.query """

from dicomstack.query import Query, Selector


def test_selector_class():
    assert Selector().key == ()

    assert Selector("a").key == ("a",)
    assert Selector().a.key == ("a",)

    assert Selector("a", 0).key == ("a", 0)
    assert Selector().a[0].key == ("a", 0)

    assert len(Selector()) == 0
    assert len(Selector("a")) == 1
    assert len(Selector("a", 1)) == 2


def test_query_class():

    query1 = Selector("a") == 1
    query2 = Selector("b") < 2
    assert isinstance(query1, Query)
    assert isinstance(query2, Query)
    assert isinstance(query1 | query2, Query)
    assert isinstance(query1 & query2, Query)
    assert isinstance(~query1, Query)

    class rget:
        """getter functor"""

        def __init__(self, db):
            self.db = db

        def __call__(self, selector):
            value = self.db
            for item in selector.keys:
                value = value[item]
            return value

    db = [{"a": 1, "b": [1, 2], "c": "foobar"}, {"a": 2, "b": [1, 3], "c": "foobaz"}]
    getters = [rget(row) for row in db]

    query1 = Selector("a") == 1
    assert [True, False] == list(map(query1, getters))

    query2 = Selector("c") == "foobaz"
    assert [False, True] == list(map(query2, getters))
    assert [False, False] == list(map(query1 & query2, getters))
    assert [True, True] == list(map(query1 | query2, getters))
    assert [True, False] == list(map(~query2, getters))

    sel = Selector()
    assert [True, False] == list(map(sel.a < 2, getters))
    assert [True, True] == list(map(sel.a <= 2, getters))
    assert [False, True] == list(map(sel.a > 1, getters))
    assert [True, True] == list(map(sel.a >= 1, getters))

    assert [True, False] == list(map(sel.c != "foobaz", getters))
    assert [True, True] == list(map(sel.c.startswith("foo"), getters))
    assert [False, True] == list(map(sel.c.endswith("baz"), getters))

    assert [True, False] == list(map(sel.a.is_in([1, 3]), getters))
    assert [False, True] == list(map(Selector("a").not_in([1, 3]), getters))

    assert [False, False] == list(map(sel.b.is_null(), getters))
    assert [True, False] == list(map(sel.b.contains(2), getters))
    assert [True, False] == list(map(sel.b.intersect([2, 4]), getters))
    assert [True, False] == list(map(sel.b.subset([1, 2, 4]), getters))
    assert [False, True] == list(map(sel.b.superset([3]), getters))
    assert [True, False] == list(map(sel.b[1] == 2, getters))
    assert [False, True] == list(map(sel.c[-1] == "z", getters))

    assert [True, True] == list(map(sel.c.regex("fooba.+"), getters))

    # nested selector
    db = {"a": {"b": [0, "foo"]}}
    getter = rget(db)
    assert (Selector("a", "b", 1) == "foo").execute(getter)
    # same thing:
    sel = Selector()
    assert (sel.a.b[1] == "foo").execute(getter)

    # all and none Query
    db = {'a': 1, 'b': 2}
    query1 = Selector("a") == 1
    getter = rget(db)

    qall = Query(True)
    assert qall.execute(getter) is True
    qnone = Query(False)
    assert qnone.execute(getter) is False
    assert (qall & qnone).execute(getter) is False
    assert (qall | qnone).execute(getter) is True
    assert (~qall).execute(getter) is False
    assert (~qnone).execute(getter) is True
    assert (~qall).execute(getter) is False

    assert (query1 & qall).execute(getter) is True
    assert (~query1 | qall).execute(getter) is True
    assert (query1 | qnone).execute(getter) is True
    assert (query1 & qnone).execute(getter) is False
