from pydantic import BaseModel

class ArticleData(BaseModel):
    text: str
    url: str

class ExplainData(BaseModel):
    text: str

class ImageData(BaseModel):
    image_url: str
