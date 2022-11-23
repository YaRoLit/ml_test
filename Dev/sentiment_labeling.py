import pandas as pd
from transformers import pipeline

df = pd.read_csv('https://raw.githubusercontent.com/YaRoLit/ml_test/b6fdcf22e4822b10f72a179c9ab3661b4cf40334/chat.csv',
                 encoding='utf-16',
                 sep='\t',
                 header=None,
                 names=['date', 'id', 'text'])

classifier = pipeline('sentiment-analysis',
                      model='blanchefort/rubert-base-cased-sentiment',
                      truncation=True,
                      max_length=512)

sentiment_labels = []

for text in df.text.values:
    sentiment = classifier(str(text))
    sentiment_labels.append(sentiment[0]['label'])

df.insert(df.shape[1],
          'sentiment', 
          sentiment_labels)

df.to_csv('Data/chat_sentiment.csv', 
          sep='\t',
          index=False,
          encoding='utf-8')
