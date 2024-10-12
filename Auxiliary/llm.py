import os
import re
import json
import faiss
import PyPDF2
import requests
import numpy as np

from Auxiliary import config
from Auxiliary.DataBase import operations
from sentence_transformers import SentenceTransformer


os.environ['TOGETHER_API_KEY'] = '21d5552e22479067e0e6010f2a0f2a07ac777cd1fe42cb3c612f5faf97e310da'


class LLMModel:
    def __init__(self):
        assert os.path.exists(config.Paths.LearningResources), "Отсутствует база знаний"

        self.model = SentenceTransformer('BAAI/bge-m3', device="cuda")

        # Создание эмбеддингов для нормативных документов
        self.chunks = self.text_split(self.pdf_to_text(config.Paths.LearningResources))
        self.embeddings = self.model.encode(self.chunks)

        # Инициализация индекса FAISS для поиска по документам
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        self.questions, self.answers, self.qna_embeddings = operations.get_QnA()

        self.qna_index = None
        if self.qna_embeddings.shape[0]:
            self.qna_index = faiss.IndexFlatL2(self.qna_embeddings.shape[1])
            self.qna_index.add(self.qna_embeddings)

    def get_embedding(self, text):
        return self.model.encode([text])[0]

    def get_similar_qna(self, question, k=3):
        # Если у нас нет базы данных вопросов и ответов
        if self.qna_index is None:
            return

        # Получение эмбеддинга для текущего вопроса
        question_embedding = self.get_embedding(question)

        # Поиск похожих вопросов
        dist, idx = self.qna_index.search(np.expand_dims(question_embedding, axis=0), k)

        # Возвращаем топ-k вопросов и их ответы
        similar_qna = [(self.questions[i], self.answers[i]) for i in idx.flatten()]
        return similar_qna

    def get_similar_chunks(self, question, k=3):
        # Получение эмбеддинга для вопроса
        question_embedding = self.get_embedding(question)

        # Поиск похожих частей нормативных документов
        dist, idx = self.index.search(np.expand_dims(question_embedding, axis=0), k)

        # Возвращаем топ-k частей документов
        similar_chunks = [self.chunks[i] for i in idx.flatten()]
        return similar_chunks

    def generate_prompt(self, question):
        # Найти похожие части нормативных документов
        similar_chunks = self.get_similar_chunks(question)

        # Найти похожие вопросы и ответы
        similar_qna = self.get_similar_qna(question)

        # Формирование промпта для модели на основе найденных частей документов
        sources_text = ' \n\n '.join([f'ИСТОЧНИК {i + 1}: {chunk}' for i, chunk in enumerate(similar_chunks)])

        # Добавляем похожие вопросы и ответы в промпт
        qna_text = ''
        if similar_qna is not None:
            qna_text = ' \n\n '.join(
                [f'ПОХОЖИЙ ВОПРОС {i + 1}: {q} \nОТВЕТ: {a}' for i, (q, a) in enumerate(similar_qna)])

        # Окончательный промпт с инструкцией
        prompt = (
            f"Вы помощник по вопросам, связанным с ОАО 'РЖД'. "
            f"Используйте только информацию, содержащуюся в источниках после слова 'ТЕКСТ' "
            f"и в разделе с 'ПОХОЖИМИ ВОПРОСАМИ' и 'ОТВЕТАМИ' после слов 'ПОХОЖИЙ ВОПРОС' и 'ОТВЕТ'. "
            f"Ответьте на вопрос, который следует за словом 'ВОПРОС'. "
            f"Ответ должен быть на русском языке, без использования нецензурной лексики и оскорбительных выражений. "
            f"Он должен быть вежливым, корректным и подан в одном абзаце. "
            f"Ссылайтесь на пункты, используя конкретные примеры, например: 'из пункта 35.1 следует'. "
            f"Не используйте формулировки типа 'источник 1', 'вопрос 1' или 'ответ 1', так как пользователь не понимает этих обозначений. "
            f"Общайтесь на 'вы', в деловом стиле, но дружелюбно. Не забудьте поздороваться. "
            f"Если вопрос является неуместным, оскорбительным или бессмысленным в контексте вашей роли как помощника по вопросам связанным с ОАО 'РЖД', "
            f"объясните кратко, почему он некорректен, и не давайте никакого дополнительного ответа или гипотетических разъяснений. "
            f"Просто укажите: 'Ваш вопрос некорректен в рамках обсуждаемых тем', если это применимо. "
            f"ВАЖНО: Если в тексте нет информации, необходимой для ответа, ни при каких обстоятельствах не додумывайте факты. "
            f"Если информации недостаточно, дайте ответ: 'У меня недостаточно информации'.\n"
            f"ТЕКСТ:\n{sources_text}\n"
            f"ПОХОЖИЕ ВОПРОСЫ И ОТВЕТЫ:\n{qna_text}\n"
            f"ВОПРОС:\n{question}"
        )

        return prompt

    def __call__(self, question):
        prompt = self.generate_prompt(question)

        data = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "max_tokens": 2048,
            "prompt": f"[INST] {prompt} [/INST]",
            "temperature": 0.4,
            "top_p": 0.7,
            "top_k": 5,
            "repetition_penalty": 1,
            "stop": ["[/INST]", "</s>"]
        }

        # Получение ответа
        response_text = self.llm_request(data)

        # Формирование списка источников
        # similar_chunks = self.get_similar_chunks(question)
        # sources_text = '\n\n'.join([f'ИСТОЧНИК {i + 1}:\n{chunk}' for i, chunk in enumerate(similar_chunks)])

        # Возвращаем полный ответ с источниками
        # full_response = f"{response_text}\n\nИСТОЧНИКИ ДАННЫХ:\n{sources_text}"

        return response_text

    def is_question_inappropriate(self, question):
        # Промпт для проверки адекватности вопроса
        prompt = (
            f"Оцените следующий вопрос с точки зрения его адекватности и осмысленности в рамках того, что вы помощник по вопросам, связанным с ОАО 'РЖД'.: '{question}'. "
            f"Если вопрос неуместный, оскорбительный или бессмысленный, ответьте 'да'. Если вопрос осмысленный и адекватный, ответьте 'нет'."
        )

        data = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "max_tokens": 50,
            "prompt": f"[INST] {prompt} [/INST]",
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 5,
            "repetition_penalty": 1,
            "stop": ["[/INST]", "</s>"]
        }

        result = self.llm_request(data)

        return result.lower().startswith('да')

    def record_qna(self, question, answer):
        # Обновление списков вопросов и ответов
        self.questions.append(question)
        self.answers.append(answer)

        # Создание эмбеддинга для нового вопроса
        question_embedding = self.get_embedding(question)

        if self.qna_index is None:
            self.qna_index = faiss.IndexFlatL2(question_embedding.shape[0])

        # Добавление нового эмбеддинга в FAISS индекс
        self.qna_index.add(question_embedding)

    @staticmethod
    def llm_request(data):
        # Подготовка и отправка запроса к API (использование модели Mixtral)
        endpoint = 'https://api.together.xyz/v1/chat/completions'
        response = requests.post(endpoint, json=data, headers={"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"})

        # Получение ответа и анализ результата
        result = dict(json.loads(response.content))['choices'][0]['message']['content'].strip()

        return result

    @staticmethod
    def pdf_to_text(pdf_path):
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ''

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text

    @staticmethod
    def text_split(text):
        # Регулярное выражение для поиска номеров пунктов, начинающихся с новой строки
        pattern = r"(?<=\n)(\d+\.\d+)\."

        # Разбиваем текст по номерам пунктов, которые идут с новой строки
        split_text = re.split(pattern, text)

        # Результирующий список
        result = []

        # Обрабатываем части текста
        for i in range(1, len(split_text), 2):
            # Восстанавливаем пункт
            num = split_text[i].strip()
            content = split_text[i + 1].strip()
            result.append(f"{num}. {content}")

        return result
