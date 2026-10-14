"""Request/response contracts for the inference API.

Field bounds mirror ``credit_default.validation`` (a test keeps the two in sync), so a
request the API accepts is always a row the training-time validator would accept.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

PayStatus = Annotated[
    int,
    Field(ge=-2, le=8, description="-2 no use, -1 paid in full, 0 revolving, 1-8 months late"),
]
BillAmount = Annotated[int, Field(ge=-10_000_000, le=10_000_000)]
PayAmount = Annotated[int, Field(ge=0, le=10_000_000)]


class CreditApplication(BaseModel):
    """One customer's raw features (NT dollars; month 1 = most recent)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "limit_bal": 50000,
                "sex": 2,
                "education": 2,
                "marriage": 1,
                "age": 35,
                "pay_1": 0,
                "pay_2": 0,
                "pay_3": 0,
                "pay_4": 0,
                "pay_5": 0,
                "pay_6": 0,
                "bill_amt1": 30000,
                "bill_amt2": 28000,
                "bill_amt3": 26000,
                "bill_amt4": 24000,
                "bill_amt5": 22000,
                "bill_amt6": 20000,
                "pay_amt1": 2000,
                "pay_amt2": 2000,
                "pay_amt3": 2000,
                "pay_amt4": 2000,
                "pay_amt5": 2000,
                "pay_amt6": 2000,
            }
        },
    )

    limit_bal: int = Field(ge=1, le=10_000_000, description="credit limit")
    sex: Literal[1, 2] = Field(description="1 male, 2 female")
    education: int = Field(
        ge=0, le=6, description="1 grad school, 2 university, 3 high school, 4 other"
    )
    marriage: int = Field(ge=0, le=3, description="1 married, 2 single, 3 other")
    age: int = Field(ge=18, le=100)
    pay_1: PayStatus
    pay_2: PayStatus
    pay_3: PayStatus
    pay_4: PayStatus
    pay_5: PayStatus
    pay_6: PayStatus
    bill_amt1: BillAmount
    bill_amt2: BillAmount
    bill_amt3: BillAmount
    bill_amt4: BillAmount
    bill_amt5: BillAmount
    bill_amt6: BillAmount
    pay_amt1: PayAmount
    pay_amt2: PayAmount
    pay_amt3: PayAmount
    pay_amt4: PayAmount
    pay_amt5: PayAmount
    pay_amt6: PayAmount


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[CreditApplication] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    default_probability: float = Field(ge=0, le=1)
    predicted_default: bool
    threshold: float


class PredictResponse(Prediction):
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    n_records: int


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None = None
    detail: str | None = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    version: str
    alias: str | None
    source: Literal["registry", "local"]
    model_type: str | None
    decision_threshold: float
    run_id: str | None
    dataset_sha256: str | None
    git_commit: str | None
    metrics: dict[str, float]
    loaded_at: str
