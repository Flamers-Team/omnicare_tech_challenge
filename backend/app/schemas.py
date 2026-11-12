from pydantic import BaseModel, Field


class Confirmation(BaseModel):
    responsavel: str = Field(min_length=2, max_length=80)
