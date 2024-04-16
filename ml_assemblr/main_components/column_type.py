from pydantic import BaseModel


class ColumnType(BaseModel):
    features: list[str] = []
    labels: list[str] = []
    predictions: list[str] = []
    splitters: list[str] = []
    keys: list[str] = []
