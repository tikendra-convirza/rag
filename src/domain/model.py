from dataclasses import dataclass
from typing import List
from pathlib import Path

@dataclass(frozen=True)
class BaseDocument:
    id: str
    text: str
    score: float
    metadata:dict
    embedding:list[float]|None

@dataclass(frozen=True)
class Citation:
    document_id: str
    snippet: str

@dataclass(frozen=True)
class Answer:
    text: str
    citations: List[Citation]

@dataclass(frozen=True)
class File:
    path:Path