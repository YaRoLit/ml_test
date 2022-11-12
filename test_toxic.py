from transformers import pipeline

classifier = pipeline('text-classification',
                      model='SkolkovoInstitute/russian_toxicity_classifier')

print(classifier('Я обожаю инженерию машинного обучения!'))
