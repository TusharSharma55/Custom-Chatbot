
from pydantic import BaseModel


class InputModel(BaseModel):
    question: str = "List of Services provided by FiftyFive Technologies ?"
