from pydantic import BaseModel


class ColumnTypes(BaseModel):
    features: list[str] = []
    labels: list[str] = []
    predictions: list[str] = []
