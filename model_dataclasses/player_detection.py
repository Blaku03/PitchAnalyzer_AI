from dataclasses import dataclass, asdict, fields
from typing import Optional, Tuple, List, Type, TypeVar
import pandas as pd

T = TypeVar("T")


@dataclass
class PlayerDetection:
    frame: int
    track_id: int
    cls: str
    team: Optional[int]
    confidence: float
    bbox: Tuple[float, float, float, float]

    def to_dict(self) -> dict:
        d = asdict(self)
        x_min, y_min, x_max, y_max = d.pop("bbox")
        d.update({"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max})
        return d

    @classmethod
    def from_dict(cls: Type[T], d: dict) -> T:
        bb = (d.pop("x_min"), d.pop("y_min"), d.pop("x_max"), d.pop("y_max"))
        return cls(bbox=bb, **{f.name: d[f.name] for f in fields(cls) if f.name in d})


class PDetectionSchema:
    @staticmethod
    def to_df(dets: List[PlayerDetection]) -> pd.DataFrame:
        return pd.DataFrame([d.to_dict() for d in dets])

    @staticmethod
    def from_df(df: pd.DataFrame) -> List[PlayerDetection]:
        return [PlayerDetection.from_dict(r) for r in df.to_dict(orient="records")]
