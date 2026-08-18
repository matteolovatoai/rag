from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Di cosa parla il documento?"})
    session_id: str = Field(..., json_schema_extra={"example": "utente_01"})

class ChatResponse(BaseModel):
    answer: str