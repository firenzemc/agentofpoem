from pydantic import BaseModel, Field


class Poem(BaseModel):
    id: str
    title: str | None = None
    author: str | None = None
    language: str
    era: str | None = None
    full_text: str
    source_name: str
    source_url: str | None = None
    license: str


class Verdict(BaseModel):
    """A single swarm agent's judgement of one candidate poem against the query."""

    poem_id: str
    match: bool = False
    confidence: float = 0.0
    matched_description_aspects: list[str] = Field(default_factory=list)
    evidence_lines: list[str] = Field(default_factory=list)
    explanation: str = ""
    explanation_lang: str = ""
