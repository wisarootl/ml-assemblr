import pytest

from ml_assemblr.utils.string_case_utils import to_dummy, to_screaming_snake_case, to_snake_case


@pytest.mark.parametrize(
    "input_string, expected",
    [
        ("hello world", "HELLO_WORLD"),
        ("snake_case", "SNAKE_CASE"),
        ("This Is A Test", "THIS_IS_A_TEST"),
        ("123number", "123NUMBER"),
        ("", ""),
    ],
)
def test_to_screaming_snake_case(input_string: str, expected: str):
    assert to_screaming_snake_case(input_string) == expected


@pytest.mark.parametrize(
    "input_string, expected",
    [
        ("hello world", "hello_world"),
        ("snake_case", "snake_case"),
        ("This Is A Test", "this_is_a_test"),
        ("123number", "123number"),
        ("", ""),
    ],
)
def test_to_snake_case(input_string: str, expected: str):
    assert to_snake_case(input_string) == expected


@pytest.mark.parametrize(
    "input_string",
    [
        "hello world",
    ],
)
def test_to_dummy(input_string: str):
    assert to_dummy(input_string) == input_string
