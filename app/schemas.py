from pydantic import BaseModel
from typing import List

class HouseInput(BaseModel):
    data: List[List[float]]