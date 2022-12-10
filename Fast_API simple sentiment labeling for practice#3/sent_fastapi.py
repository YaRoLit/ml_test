from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline("sentiment-analysis",   
                      "blanchefort/rubert-base-cased-sentiment")

@app.post("/predict/")
def predict(item: Item):
    """Sentiment analysis for a text in Russian/
    Определение тональности русского текста
    """
    
    return classifier(item.text )[0]