from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Di cosa parla il documento?"})

class ChatResponse(BaseModel):
    answer: str