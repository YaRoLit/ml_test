import streamlit as st
from transformers import pipeline

@st.cache(allow_output_mutation=True)

def classifier_func(string):
    """Функция определения тональности русского текста из библиотеки Hugging Face
    """
    classifier = pipeline("sentiment-analysis",   
                      "blanchefort/rubert-base-cased-sentiment")
    return classifier(string)

def print_predictions(preds):
    """Функция вывода результата
    """
    cl = classifier_func(string)
    st.write(cl[0]['label'], cl[0]['score'])

# Выводим заголовок страницы средствами Streamlit     
st.title('Классификация тональности русского текста')
# Вызываем функцию ввода текста
string = st.text_input("Введи русский текст для классификации")
result = st.button('Классифицировать текст')
if result:
    preds =  classifier_func(string)
    st.write('**Результаты распознавания тональности текста:**')
    print_predictions(preds)
