import re


def to_screaming_snake_case(input_string: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", input_string).upper()


def to_snake_case(input_string: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", input_string).lower()
