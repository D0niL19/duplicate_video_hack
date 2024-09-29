import faiss
import numpy as np
import hashlib

class VideoDuplicateSearcherCosine:
    def __init__(self, dim):
        # Индекс для поиска по косинусной близости (внутреннее произведение)
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = {}  # Словарь для метаданных
        self.dim = dim

    def _hash_embedding(self, embedding):
        """Создание хеша для эмбеддинга для использования в качестве ключа в метаданных."""
        # Преобразуем эмбеддинг в строку и хешируем с помощью SHA256
        embedding_bytes = embedding.tobytes()
        return hashlib.sha256(embedding_bytes).hexdigest()

    def add_embedding(self, embedding, uuid, link):
        # Нормализуем эмбеддинг для косинусного поиска
        normalized_embedding = embedding / np.linalg.norm(embedding)
        # Добавляем его в FAISS индекс
        self.index.add(np.array([normalized_embedding], dtype=np.float32))

        # Храним метаданные в словаре с хешем эмбеддинга
        embedding_hash = self._hash_embedding(normalized_embedding)
        self.metadata[embedding_hash] = {'uuid': uuid, 'link': link, 'embedding': normalized_embedding}

    def search(self, query_embedding, k=5):
        # Нормализуем запрос
        normalized_query = query_embedding / np.linalg.norm(query_embedding)

        # Выполняем поиск по индексу
        distances, indices = self.index.search(np.array([normalized_query], dtype=np.float32), k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]

            # Получаем хеш эмбеддинга по индексу
            embedding_hash = list(self.metadata.keys())[idx]

            # Получаем метаданные по хешу
            metadata = self.metadata[embedding_hash]

            results.append({
                'distance': distance,
                'uuid': metadata['uuid'],
                'link': metadata['link'],
                'embedding': metadata['embedding']  # Добавляем сам эмбеддинг
            })

        return results

    def get_size(self):
        return self.index.ntotal