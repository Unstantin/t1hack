from qdrant_client import QdrantClient
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, BartForConditionalGeneration, BartTokenizer
import pandas as pd
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from TextParser import TextParser
import psycopg2
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

def add_document(bot, message):
    print("НАЧАЛО ОБРАБОТКИ")
    try:
        bot.send_message(message.chat.id, 'Загружаем файл в бд')

        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with open('received_file', 'wb') as new_file:
            new_file.write(downloaded_file)

        file_type = message.document.mime_type
        print(file_type)

        parser = TextParser()
        text = None
        if (file_type=="text/plain"):
            text = parser.parse_txt('received_file')
        if (file_type=="application/pdf"):
            text = parser.parse_pdf('received_file')
        if (file_type=="application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
            text = parser.parse_word('received_file')
        if (file_type=="text/csv"):
            text = parser.parse_csv('received_file')

        print(text)

        with open('received_file', 'wb') as new_file:
            new_file.write(downloaded_file)

        # Инициализация Qdrant клиента
        qdrant_client = QdrantClient(host="localhost", port=6333)
        # Загрузка модели и токенизатора для DPRQuestionEncoder
        tokenizer_dpr = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        model_dpr = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

        def generate_vectors(text):
            inputs = tokenizer_dpr(text, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                vector = model_dpr(**inputs).pooler_output.squeeze().numpy()
            return vector.tolist()

        # Функция для загрузки текста в Qdrant
        def upload_text_to_qdrant(collection_name, text):
            # Разбиение текста на предложения
            sentences = sent_tokenize(text)

            # Генерация векторов для каждого предложения
            vectors = [generate_vectors(sentence) for sentence in sentences]

            # Загрузка предложений в Qdrant
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    {"id": i, "vector": vector, "payload": {"text": sentence}}
                    for i, (sentence, vector) in enumerate(zip(sentences, vectors))
                ]
            )
            print("Текст успешно загружен в Qdrant.")

        upload_text_to_qdrant(getCollectionOfUser(message.from_user.id), text)

        # bot.delete_message(message.chat.id, sent_message.message_id)
        print("Векторы успешно загружены в Qdrant.")

        bot.send_message(message.chat.id, 'Файл успешно обработан.')
    except Exception as e:
        bot.reply_to(message, f'Произошла ошибка: {e}')


def getCollectionOfUser(telegramm_id):
    conn_params = {
        "dbname": "t1_hakaton",
        "user": "postgres",
        "password": "7l1282",
        "host": "localhost",
        "port": "5432"
    }

    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()

    select_query = """
        SELECT name
        FROM qdrant_db
        WHERE admin_tg_id = %(admin_tg_id)s;
    """
    cur.execute(select_query, {"admin_tg_id": telegramm_id})
    result = cur.fetchone()
    return str(telegramm_id) + "_" + result[0]