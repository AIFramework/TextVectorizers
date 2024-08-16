import os
import random
from tools import nltk_resource_download
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split

nltk_resource_download('punkt')

class BiEncoderDatasetGenerator:
    def __init__(self, tokenizer_model, vectorizer_model, language = "russian"):
      """
      Инициализирует генератор датасета для BiEncoder модели.

      Параметры:
      - tokenizer_model: Модель для токенизации текста.
      - vectorizer_model: Модель для векторизации текста.
      - language: Язык текста (по умолчанию "russian").
      """
      
        self.tokenizer = tokenizer_model
        self.vectorizer = vectorizer_model
        self.language = language

    def generate_dataset(self, folder_path, sample_size, test_size):
      """
      Генерирует датасет для BiEncoder модели.

      Параметры:
      - folder_path: Путь к папке с текстовыми файлами.
      - sample_size: Количество пар для выборки.
      - test_size: Размер тестового набора.
      """
        
        pairs = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                
                sentences = sent_tokenize(text, language = self.language)
                
                # Создаем пары "текущее предложение - следующее предложение"
                for i in range(len(sentences) - 1):
                    pairs.append((sentences[i], sentences[i + 1]))

        random.shuffle(pairs)
        
        if len(pairs) > sample_size:
            pairs = pairs[:sample_size]

        tokenized_pairs = [(self.tokenizer.encode(pair[0]), self.vectorizer.encode(pair[1])) for pair in pairs]

        # Разделяем на обучающий и тестовый наборы
        train_data, test_data = train_test_split(tokenized_pairs, test_size=test_size, random_state=42)

        return train_data, test_data