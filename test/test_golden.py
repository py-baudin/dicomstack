""" compare dicomstack's outputs with the golden snapshot (see make_golden.py) """

import os
import pytest
import golden_lib


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    return golden_lib.make_fixtures(str(tmp_path_factory.mktemp("golden")))


@pytest.mark.parametrize("name", golden_lib.FIXTURES)
def test_golden(paths, name):
    if not os.path.isfile(golden_lib.golden_path(name)):
        # golden data is not versioned: run test/make_golden.py with the reference code
        pytest.skip("no golden snapshot: run test/make_golden.py")
    expected = golden_lib.load(name)
    actual = golden_lib.snapshot({name: paths[name]})[name]
    # round-trip through json for a like-for-like comparison
    actual = golden_lib.json.loads(golden_lib.json.dumps(actual))
    differences = golden_lib.diff(expected, actual)
    assert not differences, "\n".join(differences)
