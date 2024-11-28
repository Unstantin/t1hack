import telebot
import psycopg2
import requests
import os
import PyPDF2
from docx import Document
import pandas as pd
import spacy
from qdrant_client import QdrantClient
# import insertData from createData
import pandas as pd
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from qdrant_client import QdrantClient
import numpy as np
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, BartForConditionalGeneration, BartTokenizer
from qdrant_client import QdrantClient
import TextParser
from add_document import add_document
from qdrant_client.http.models import VectorParams, Distance

bot = telebot.TeleBot('7590479270:AAHRnbclsYfJgBylK8hr5tpgikyiyViJEHA')


@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, 'Привет!')


@bot.message_handler(commands=['create_db'])
def start_create_db(message):
    bot.send_message(chat_id=message.chat.id, text="Выберите название базе знаний")
    bot.register_next_step_handler_by_chat_id(message.chat.id, create_db)


def create_db(message):
    conn_params = {
        "dbname": "t1_hakaton",
        "user": "postgres",
        "password": "7l1282",
        "host": "localhost",
        "port": "5432"
    }

    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()

    data = {
        "admin_tg_id": message.from_user.id,
        "name": message.text
    }

    insert_query = """
    INSERT INTO qdrant_db (id, admin_tg_id, name)
    VALUES (DEFAULT, %(admin_tg_id)s, %(name)s)
    """
    cur.execute(insert_query, data)
    conn.commit()

    cur.close()
    conn.close()

    qdrant_client = QdrantClient(host="localhost", port=6333)
    db_name = message.from_user.id + '_' + message.text
    if not qdrant_client.collection_exists(db_name):
        qdrant_client.create_collection(
            collection_name=db_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        bot.send_message(chat_id=message.chat.id, text="База знаний создана")
    else:
        bot.send_message(chat_id=message.chat.id, text="Имя уже занято, выберите другое")
        bot.register_next_step_handler_by_chat_id(message.chat.id, create_db)


# Обработчик файлов
#@bot.message_handler(content_types=['document'])
def handle_document(message):
    add_document(bot, message)
    bot.send_message(message.chat.id, "ЗАГРУЖЕНО")


@bot.message_handler(commands=['ask'])
def ask(message):
    new_m = bot.send_message(message.chat.id, 'Собираю данные')
    # Инициализация Qdrant клиента
    qdrant_client = QdrantClient(host="localhost", port=6333)

    # Загрузка модели и токенизатора для DPRQuestionEncoder
    tokenizer_dpr = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    model_dpr = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    # Загрузка модели и токенизатора для BART
    tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    # Функция для получения релевантных данных
    def retrieve_relevant_data(query, collection_name):
        inputs = tokenizer_dpr(query, return_tensors="pt")
        with torch.no_grad():
            query_vector = model_dpr(**inputs).pooler_output.squeeze().numpy()

        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5
        )

        relevant_descriptions = []
        for hit in search_result:
            if "text" in hit.payload:
                relevant_descriptions.append(hit.payload["text"])
            else:
                print(f"Warning: Missing 'text' key in payload: {hit.payload}")

        return relevant_descriptions

    # Функция для генерации ответа с использованием BART
    def generate_response(query, relevant_data):
        prompt = f"Query: {query}\n {relevant_data}\nGenerate a response based on the relevant data:\nResponse:"
        inputs = tokenizer_bart(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model_bart.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
        response = tokenizer_bart.decode(outputs[0], skip_special_tokens=True)

        # Обработка вывода для улучшения читаемости
        response = response.replace("Response:", "").strip()
        response = response.replace("[", "").replace("]", "").strip()

        return response

    # Пример использования функций
    collection_name = "my_collection"
    query = message.text
    relevant_data = retrieve_relevant_data(query, collection_name)
    response = generate_response(query, relevant_data)
    bot.delete_message(message.chat.id, new_m.message_id)
    bot.send_message(message.chat.id, response)

    '''nlp = spacy.load("ru_core_news_sm")

    doc = nlp(message.text)

    keywords = [token.text for token in doc if not token.is_stop and not token.is_punct]
    keywords_token = [token for token in doc if not token.is_stop and not token.is_punct]

    bot.send_message(message.chat.id, ", ".join(keywords))

    keyword_vectors = {token.text: token.vector for token in keywords_token}

    for word, vector in keyword_vectors.items():
        print(f"Ключевое слово: {word}")
        print(f"Векторное представление: {vector}\n")

    # requestForDatabase(keyword_vectors)'''


@bot.message_handler(commands=['load_file'])
def load_file(message):
    bot.send_message(message.chat.id, "Отправляйте файлы")
    bot.register_next_step_handler_by_chat_id(message.chat.id, handle_document)


# Запуск бота
bot.polling(none_stop=True)
