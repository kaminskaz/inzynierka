from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class ResponseSchema(BaseModel):
    answer: str = Field(..., description="The model's answer to the question.")
    confidence: float = Field(..., description="The model's confidence in its answer, ranging from 0.0 to 1.0.")
    rationale: str = Field(..., description="The model's rationale for its answer.")

class BPResponseSchema(BaseModel):
    left_side_rule: str = Field(..., description="The model's description of the left side rule.")
    right_side_rule: str = Field(..., description="The model's description of the right side rule.")
    confidence: float = Field(..., description="The model's confidence in its answer, ranging from 0.0 to 1.0.")
    rationale: str = Field(..., description="The model's rationale for its answer.")


class DescriptionResponseSchema(BaseModel):
    description: str = Field(
        ..., description="A detailed description of the provided image content."
    )

class BPDescriptionResponseSchemaContrastive(BaseModel):
    description_left: str = Field(
        ..., description="A detailed description of the left provided image content."
    )
    description_right: str = Field(
        ..., description="A detailed description of the right provided image content."
    )

# for now not in use
class SimilarityResponseSchema(BaseModel):
    similarity_label: str = Field(
        ...,
        description="A similarity score indicating how similar the two inputs are.",
    )

# strictly constrain the LLM's output choices
class EvaluationLabel(str, Enum):
    RIGHT = "Right"
    SOMEWHAT_RIGHT = "Somewhat right"
    UNCLEAR = "Unclear"
    SOMEWHAT_WRONG = "Somewhat wrong"
    WRONG = "Wrong"

# bongard similarity evaluation schema
class BongardEvaluationSchema(BaseModel):
    reasoning: str = Field(
        ...,
        description="A one-sentence reasoning explaining the decision based on the rubric."
    )
    
    similarity_label: EvaluationLabel = Field(
        ...,
        description="The specific categorical label representing the similarity."
    )

class GeneralEnsembleSchema(BaseModel):
    final_answer: str = Field(
        ..., description="The final answer chosen by the ensemble method."
    )
    rationale: str = Field(
        ..., description="The rationale behind the ensemble's final answer."
    )