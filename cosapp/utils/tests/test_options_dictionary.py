"""
Copyright (c) 2016-2018, openmdao.org

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This module comes from OpenMDAO 2.2.0. It was slightly modified for CoSApp integration.
"""
import pytest

from cosapp.systems import System
from cosapp.utils.options_dictionary import OptionsDictionary


def check_even(name, value):
    if value % 2 != 0:
        raise ValueError(
            f"Option '{name}' with value {value} is not an even number."
        )


class MyComp(System):
    pass


@pytest.fixture
def options():
    options = OptionsDictionary()
    my_comp = MyComp("my_comp")

    options.declare("test", values=["a", "b"], desc="Test integer value")
    options.declare("flag", default=False, dtype=bool)
    options.declare("comp", default=my_comp, dtype=System)
    options.declare("undefined")
    options.declare(
        "long_desc",
        dtype=str,
        desc="This description is long and verbose, so it "
        "takes up multiple lines in the options table.",
    )
    return options


def test_OptionsDictionary_reprs(options):
    assert repr(options) == repr(options._dict)

    print(options.__str__(width=83))
    assert options.__str__(width=83) == "\n".join(
        [
            "========= ================ ================= ================ =====================",
            "Option    Default          Acceptable Values Acceptable Types Description          ",
            "========= ================ ================= ================ =====================",
            "comp      my_comp - MyComp N/A               ['System']                            ",
            "flag      False            N/A               ['bool']                              ",
            "long_desc **Required**     N/A               ['str']          This description is l",
            "                                                              ong and verbose, so i",
            "                                                              t takes up multiple l",
            "                                                              ines in the options t",
            "                                                              able.",
            "test      **Required**     ['a', 'b']        N/A              Test integer value   ",
            "undefined **Required**     N/A               N/A                                   ",
            "========= ================ ================= ================ =====================",
        ]
    )

    # if the table can't be represented in specified width, then we get the full width version
    long_desc = "This description is long and verbose, so it takes up multiple lines in the options table."
    assert options.__str__(width=40) == "\n".join(
        [
            "========= ================ ================= ================ ====================="
            "==================================================================== ",
            "Option    Default          Acceptable Values Acceptable Types Description          "
            "                                                                     ",
            "========= ================ ================= ================ ====================="
            "==================================================================== ",
            "comp      my_comp - MyComp N/A               ['System']                            "
            "                                                                     ",
            "flag      False            N/A               ['bool']                              "
            "                                                                     ",
            "long_desc **Required**     N/A               ['str']          This description is l"
            "ong and verbose, so it takes up multiple lines in the options table. ",
            "test      **Required**     ['a', 'b']        N/A              Test integer value   "
            "                                                                     ",
            "undefined **Required**     N/A               N/A              " + " " * (len(long_desc) + 1),
            "========= ================ ================= ================ ====================="
            "==================================================================== ",
        ]
    )


def test_OptionsDictionary_type_checking():
    opt = OptionsDictionary()
    opt.declare("test", dtype=int, desc="Test integer value")

    opt["test"] = 1
    assert opt["test"] == 1

    class_or_type = "class"
    expected_msg = r"Value \(''\) of option 'test' has type of \(<{} 'str'>\), but expected type \(<{} 'int'>\)\.".format(
        class_or_type, class_or_type
    )
    with pytest.raises(TypeError, match=expected_msg):
        opt["test"] = ""

    # make sure bools work
    opt.declare("flag", default=False, dtype=bool)
    assert opt["flag"] == False
    opt["flag"] = True
    assert opt["flag"] == True


def test_OptionsDictionary_allow_none():
    opt = OptionsDictionary()
    opt.declare("test", dtype=int, allow_none=True, desc="Test integer value")
    opt["test"] = None
    assert opt["test"] == None


def test_OptionsDictionary_type_and_values():
    opt = OptionsDictionary()
    # Test with only type_
    opt.declare("test1", dtype=int)
    opt["test1"] = 1
    assert opt["test1"] == 1

    # Test with only values
    opt.declare("test2", values=["a", "b"])
    opt["test2"] = "a"
    assert opt["test2"] == "a"

    # Test with both type_ and values
    with pytest.raises(
        Exception, match="'dtype' and 'values' were both specified for option 'test3'."
    ):
        opt.declare("test3", dtype=int, values=["a", "b"])


def test_OptionsDictionary_isvalid():
    opt = OptionsDictionary()
    opt.declare("even_test", dtype=int, check_valid=check_even)
    opt["even_test"] = 2
    opt["even_test"] = 4

    with pytest.raises(
        ValueError, match="Option 'even_test' with value 3 is not an even number."
    ):
        opt["even_test"] = 3


def test_OptionsDictionary_unnamed_args():
    opt = OptionsDictionary()
    # KeyError ends up with an extra set of quotes.
    with pytest.raises(
        KeyError,
        match="\"Option 'test' cannot be set because it has not been declared.\"",
    ):
        opt["test"] = 1


def test_OptionsDictionary_contains(options):
    assert "test" in options
    assert "flag" in options
    assert "comp" in options
    assert "long_desc" in options
    assert "undefined" in options
    assert "undeclared" not in options


def test_OptionsDictionary_keys(options):
    assert set(options.keys()) == {
        "test",
        "flag",
        "comp",
        "undefined",
        "long_desc",
    }
    assert set(options.keys()) == set(options)
    assert len(options.keys()) == len(options)


def test_OptionsDictionary_values(options):
    with pytest.raises(RuntimeError, match="required but has not been set"):
        values = list(options.values())
    # Define required options
    options['test'] = 'b'
    options['undefined'] = 0
    options['long_desc'] = 'abracadabra'

    values = list(options.values())
    assert len(values) == len(options)

    for key, value in zip(options, options.values()):
        assert value == options[key], f"key: {key}"


def test_OptionsDictionary_items(options):
    with pytest.raises(RuntimeError, match="required but has not been set"):
        list(options.items())
    # Define required options
    options['test'] = 'b'
    options['undefined'] = 0
    options['long_desc'] = 'abracadabra'

    for key, value in options.items():
        assert value == options[key], f"key: {key}"
    
    items = list(options.items())
    assert len(items) == len(options)
    assert items == list(zip(options.keys(), options.values()))


def test_OptionsDictionary_update():
    opt = OptionsDictionary()
    opt.declare("test", default="Test value", dtype=object)

    obj = object()
    opt.update({"test": obj})
    assert opt["test"] is obj


def test_OptionsDictionary_update_extra():
    opt = OptionsDictionary()
    # KeyError ends up with an extra set of quotes.
    with pytest.raises(
        KeyError,
        match="\"Option 'test' cannot be set because it has not been declared.\"",
    ):
        opt.update({"test": 2})


def test_OptionsDictionary_get_missing():
    opt = OptionsDictionary()
    with pytest.raises(KeyError, match="\"Option 'missing' cannot be found\""):
        opt["missing"]


def test_OptionsDictionary_get_default():
    opt = OptionsDictionary()
    obj_def = object()
    obj_new = object()

    opt.declare("test", default=obj_def, dtype=object)

    assert opt["test"] is obj_def

    opt["test"] = obj_new
    assert opt["test"] is obj_new


def test_OptionsDictionary_values():
    opt = OptionsDictionary()
    obj1 = object()
    obj2 = object()
    opt.declare("test", values=[obj1, obj2])

    opt["test"] = obj1
    assert opt["test"] is obj1

    with pytest.raises(
        ValueError,
        match=(
            r"Value \(<object object at 0x[0-9A-Fa-f]+>\) of option 'test' is not one of \[<object object at 0x[0-9A-Fa-f]+>,"
            r" <object object at 0x[0-9A-Fa-f]+>\]."
        ),
    ):
        opt["test"] = object()


def test_OptionsDictionary_read_only():
    opt = OptionsDictionary()
    opt = OptionsDictionary(read_only=True)
    opt.declare("permanent", 3.0)

    with pytest.raises(KeyError, match="Tried to set read-only option 'permanent'."):
        opt["permanent"] = 4.0


def test_OptionsDictionary_bounds():
    opt = OptionsDictionary()
    opt.declare("x", default=1.0, lower=0.0, upper=2.0)

    with pytest.raises(
        ValueError,
        match=r"Value \(3\.0\) of option 'x' exceeds maximum allowed value 2\.0",
    ):
        opt["x"] = 3.0

    with pytest.raises(
        ValueError,
        match=r"Value \(-3\.0\) of option 'x' is less than minimum allowed value 0\.0",
    ):
        opt["x"] = -3.0


def test_OptionsDictionary_undeclare():
    opt = OptionsDictionary()
    # create an entry in the dict
    opt.declare("test", dtype=int)
    opt["test"] = 1

    # prove it's in the dict
    assert opt["test"] == 1

    # remove entry from the dict
    opt.undeclare("test")

    # prove it is no longer in the dict
    with pytest.raises(KeyError, match="Option 'test' cannot be found"):
        opt["test"]


def test_OptionsDictionary_len():
    opt = OptionsDictionary()
    opt.declare("foo", 3.0)
    opt.declare("bar", -1, dtype=int)
    assert len(opt) == 2


def test_OptionsDictionary_clear():
    opt = OptionsDictionary()
    opt.declare("foo", 3.0)
    opt.declare("bar", -1, dtype=int)
    assert len(opt) == 2
    opt.clear()
    assert len(opt) == 0


def test_OptionsDictionary_clear_read_only():
    opt = OptionsDictionary(read_only=True)
    opt.declare("permanent", 3.0)
    assert len(opt) > 0
    with pytest.raises(KeyError):
        opt.clear()
