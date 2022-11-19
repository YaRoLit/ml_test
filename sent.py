from transformers import pipeline

classifier = pipeline('sentiment-analysis',
                      model='blanchefort/rubert-base-cased-sentiment')

output = classifier(input())

print(output)
