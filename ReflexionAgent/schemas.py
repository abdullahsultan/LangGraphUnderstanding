from typing import List
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="The reflection on the initial answer.")
    search_qureries: List[str] = Field(description="1-3 search queries for searching improvement to address the critiquue of your current answer.")
    