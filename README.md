# 4PANDAS: Видео Дубликаты API

Этот проект предоставляет сервис для поиска дубликатов видео. Сервис использует современные технологии, такие как Hugging Face, ONNX, Pytorch, Triton, FAISS и FastAPI, для извлечения признаков аудио и видео из входных данных и их дальнейшей обработки для определения дубликатов.

## Технологический стек

Для реализации решения использовались следующие технологии и инструменты:
- Hugging Face — для моделей машинного обучения.
- ONNX — для оптимизации и переноса моделей.
- Pytorch — для работы с нейронными сетями и обучением моделей.
- Triton — для высокопроизводительного инференса моделей.
- FAISS — для быстрого поиска по векторным представлениям.
- FastAPI — для создания API сервиса.

## Архитектура решения

1. API принимает ссылку на видео.
2. Видео скачивается в виде потока байтов и передаётся на Triton-сервер через gRPC.
3. На Triton сервере видео обрабатывается с использованием двух экстракторов:
   - Audio Feature Extractor — для извлечения аудио признаков.
   - Video Feature Extractor — для извлечения видео признаков.
4. Извлечённые признаки передаются в систему поиска, основанную на FAISS, для определения схожести с ранее загруженными видео.

## Метрики

Обработка на CPU:

- F1-мера - 0.82
- Среднее время обработки видео: 1100 мс на видеофайл

[//]: # (- Пропускная способность сервиса: до 1000 видеофайлов в минуту)

## Установка и запуск

### Шаги установки

1. Загрузка Docker-образов

   Перед запуском необходимо загрузить нужные Docker-образы для Triton и моделей машинного обучения:
```
docker pull nvcr.io/nvidia/tritonserver:24.08-py3
```

2. Загрузить веса моделей

```commandline
mkdir -p triton/model_repository_main/audio_embedding/1 && wget -O "triton/model_repository_main/audio_embedding/1/model.onnx" https://storage.yandexcloud.net/weights/model.onnx
```
```commandline
mkdir -p triton/model_repository_main/video_embedding/1 && wget -O "triton/model_repository_main/video_embedding/1/model.onnx" https://storage.yandexcloud.net/weights/video.onnx
```

```commandline
mkdir -p triton/model_repository_main/ensemble_model/1
```

3. Запуск с Docker Compose

   Настройте и запустите сервисы с помощью Docker Compose из корня проекта.
```
docker compose up --build
```

   

### Пример запроса к API

После запуска сервис будет доступен на 89.169.154.167:8080. Пример запроса на определение дубликата видео:

curl -X POST "89.169.154.167:8080/check-video-duplicate" \
-H "Content-Type: application/json" \
-d '{"url": "https://s3.ritm.media/yappy-db-duplicates/video_id.mp4}'

Ответ будет содержать результат сравнения видео с уже существующими в базе данных.
\
Пример результата:
```
{
    "is_duplicate": false,
    "duplicate_for": ""
}
```
```
{
    "is_duplicate": true,
    "duplicate_for": "22d891cc-563a-48c9-9b6e-368829598e91"
}
```
## Контакты

Если у вас возникли вопросы или предложения, вы можете связаться с нами по следующим контактам:
- Telegram: [@D0niL19](https://t.me/D0niL19) [@Wortex04](https://t.me/Wortex04) [@AzamatSibgatullin](https://t.me/AzamatSibgatullin)
