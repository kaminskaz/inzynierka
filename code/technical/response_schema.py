from pydantic import BaseModel, Field
from typing import List

class ResponseSchema(BaseModel):    
    answer: str = Field(..., description="The model's answer to the question.")
    confidence: float = Field(..., description="The model's confidence in its answer, ranging from 0.0 to 1.0.")
    rationale: str = Field(..., description="The model's rationale for its answer.")

class DescriptionResponseSchema(BaseModel):
    description: str = Field(..., description="A detailed description of the provided image content.")

class BPDescriptionResponseSchemaContrastive(BaseModel):
    description_left: str = Field(..., description="A detailed description of the left provided image content.")
    description_right: str = Field(..., description="A detailed description of the right provided image content.")

class SimilarityResponseSchema(BaseModel):
    similarity_score: float = Field(..., description="A similarity score between 0.0 and 1.0 indicating how similar the two inputs are.")
