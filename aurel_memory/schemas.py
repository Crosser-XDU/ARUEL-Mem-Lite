from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .media import decode, MAX_IMAGES


class ImageURL(BaseModel):
    url: str
    detail: Literal["low","high","auto"] | None = None

    @field_validator("url")
    @classmethod
    def valid(cls, v):
        decode(v)
        return v


class TextPart(BaseModel):
    type: Literal["text"]
    text: str = Field(min_length=1)


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


Part = Annotated[TextPart | ImagePart, Field(discriminator="type")]
Content = str | list[Part]


def media_bytes(content):
    if isinstance(content, str):
        return 0
    total = 0
    for p in content:
        if p.type == "image_url":
            b = p.image_url.url.split(",",1)[1]
            total += len(b)*3//4 - (len(b)-len(b.rstrip("=")))
    return total


def nonempty(v):
    if not v or (isinstance(v,str) and not v.strip()):
        raise ValueError("content must not be empty")
    return v


class Message(BaseModel):
    role: Literal["user","assistant"]
    content: Content
    timestamp: int | None = Field(default=None, ge=0, le=253402300799000)
    _nonempty = field_validator("content")(nonempty)


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    request_id: str = Field(min_length=1,max_length=512)
    user_id: str = Field(min_length=1,max_length=512)
    session_id: str = Field(min_length=1,max_length=512)
    messages: list[Message] = Field(min_length=1,max_length=1024)

    @model_validator(mode="after")
    def limits(self):
        if sum(media_bytes(x.content) for x in self.messages) > MAX_IMAGES:
            raise ValueError("Add image total exceeds 30 MiB")
        return self


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str = Field(min_length=1,max_length=512)
    query: Content
    options: list[str] | None = None
    top_k: int = Field(ge=1,le=100)
    _nonempty = field_validator("query")(nonempty)

    @model_validator(mode="after")
    def limits(self):
        if media_bytes(self.query) > MAX_IMAGES:
            raise ValueError("query image total exceeds 30 MiB")
        return self
