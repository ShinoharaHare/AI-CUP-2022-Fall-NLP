import json
from collections import ChainMap
from dataclasses import dataclass, replace
from typing import List, Literal, Mapping, Optional, Tuple

Stance = Literal['AGREE', 'DISAGREE']


def get_annotations(cls) -> ChainMap:
    return ChainMap(*(c.__annotations__ for c in cls.__mro__ if '__annotations__' in c.__dict__))


class ItemUtilsMixin:
    __mapping__: Mapping[str, str] = {}

    def __getitem__(self, name: str):
        return getattr(self, name)
    
    @classmethod
    def from_dict(cls, d: dict):
        dd = {}
        for k, v in d.items():
            if k in cls.__mapping__:
                k = cls.__mapping__[k]
            
            if k in get_annotations(cls):
                dd[k] = v
        return cls(**dd)

    @classmethod
    def from_json(cls, s: str):
        d = json.loads(s)
        return cls.from_dict(d)


@dataclass
class CommonFieldsMixin:
    id: str
    q: str
    r: str
    s: Stance


@dataclass
class RawItem(CommonFieldsMixin, ItemUtilsMixin):
    __mapping__ = {'q\'': 'q_prime', 'r\'': 'r_prime'}

    q_prime: Optional[str] = None
    r_prime: Optional[str] = None

    def remove_quote(self):
        return replace(
            self,
            q=self.q.strip('"'),
            r=self.r.strip('"'),
            q_prime=self.q_prime.strip('"') if self.q_prime is not None else None,
            r_prime=self.r_prime.strip('"') if self.r_prime is not None else None
        )

    def __hash__(self):
        return hash((self.q, self.r, self.s, self.q_prime, self.r_prime))

@dataclass
class SpanItem(CommonFieldsMixin, ItemUtilsMixin):
    q_spans: List[Tuple[int, int]]
    r_spans: List[Tuple[int, int]]


@dataclass
class Spans(ItemUtilsMixin):
    q: List[Tuple[int, int]]
    r: List[Tuple[int, int]]


@dataclass
class MultiTargetItem(CommonFieldsMixin, ItemUtilsMixin):
    spans: List[Spans]

    @classmethod
    def from_dict(cls, d: dict):
        d = d.copy()
        d['spans'] = list(map(Spans.from_dict, d['spans']))
        return super().from_dict(d)
