import faiss
import numpy as np
import hashlib

class VideoDuplicateSearcherCosine:
    """
    Класс для поиска дубликатов видео на основе косинусной близости.

    Attributes:
        index (faiss.IndexFlatIP): FAISS индекс для поиска по косинусной близости.
        metadata (dict): Словарь для хранения метаданных.
        dim (int): Размерность эмбеддингов.
    """

    def __init__(self, dim):
        """
        Инициализация класса VideoDuplicateSearcherCosine.

        Args:
            dim (int): Размерность эмбеддингов.
        """
        self.index = faiss.IndexFlatIP(dim)  # Индекс для поиска по косинусной близости (внутреннее произведение)
        self.metadata = {}  # Словарь для метаданных
        self.dim = dim

    def _hash_embedding(self, embedding):
        """
        Создание хеша для эмбеддинга для использования в качестве ключа в метаданных.

        Args:
            embedding (np.ndarray): Эмбеддинг видео.

        Returns:
            str: SHA256 хеш эмбеддинга.
        """
        embedding_bytes = embedding.tobytes()
        return hashlib.sha256(embedding_bytes).hexdigest()

    def add_embedding(self, embedding, uuid, link):
        """
        Добавляет эмбеддинг видео в индекс и сохраняет его метаданные.

        Args:
            embedding (np.ndarray): Эмбеддинг видео.
            uuid (str): Уникальный идентификатор видео.
            link (str): Ссылка на видео.
        """
        normalized_embedding = embedding / np.linalg.norm(embedding)  # Нормализуем эмбеддинг для косинусного поиска
        self.index.add(np.array([normalized_embedding], dtype=np.float32))  # Добавляем его в FAISS индекс

        embedding_hash = self._hash_embedding(normalized_embedding)  # Хешируем эмбеддинг
        self.metadata[embedding_hash] = {'uuid': uuid, 'link': link, 'embedding': normalized_embedding}

    def search(self, query_embedding, k=5):
        """
        Выполняет поиск по индексу для заданного эмбеддинга запроса.

        Args:
            query_embedding (np.ndarray): Эмбеддинг запроса.
            k (int): Количество ближайших соседей для поиска.

        Returns:
            list: Список результатов поиска, содержащих расстояние, uuid, ссылку и эмбеддинг.
        """
        normalized_query = query_embedding / np.linalg.norm(query_embedding)  # Нормализуем запрос

        distances, indices = self.index.search(np.array([normalized_query], dtype=np.float32), k)  # Выполняем поиск

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]

            embedding_hash = list(self.metadata.keys())[idx]  # Получаем хеш эмбеддинга по индексу
            metadata = self.metadata[embedding_hash]  # Получаем метаданные по хешу

            results.append({
                'distance': distance,
                'uuid': metadata['uuid'],
                'link': metadata['link'],
                'embedding': metadata['embedding']  # Добавляем сам эмбеддинг
            })

        return results

    def get_size(self):
        """
        Возвращает количество эмбеддингов в индексе.

        Returns:
            int: Общее количество эмбеддингов в индексе.
        """
        return self.index.ntotal