from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .utils import VideoDuplicateSearcherCosine
import numpy as np
import requests
import tritonclient.grpc as grpcclient
import pandas as pd
import logging
import onnxruntime as ort
import torch
from torch import nn

router = APIRouter()

ort_session = ort.InferenceSession("./weights/similarity_model11.onnx")

similarity_model = nn.Sequential(
            nn.Linear(768 * 4, 768 * 2),
            nn.ELU(),
            nn.Linear(768 * 2, 768),
            nn.ELU(),
            nn.Linear(768, 1),
            nn.Sigmoid()
)
similarity_model.load_state_dict(torch.load('/content/model_last.pth', map_location=torch.device('cpu')))


# Инициализация Triton клиента и FAISS индекса
triton_client = grpcclient.InferenceServerClient(url="triton:8001")
faiss_index = VideoDuplicateSearcherCosine(1536)  # Размерность 1536 (видео) + 1536 (аудио)

# Пример эмбеддинга для теста (должен содержать правильную размерность)
# embedding_example = np.random.rand(1536)  # Размерность 1536 + 1536
# faiss_index.add_embedding(embedding_example, "example_uuid", "http://example.com/video.mp4")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Модели для запросов и ответов
class VideoLinkRequest(BaseModel):
    url: str

class VideoLinkResponse(BaseModel):
    is_duplicate: bool
    duplicate_for: str

class VideoData(BaseModel):
    created: str
    uuid: str
    link: str
    is_duplicate: bool
    duplicate_for: str
    is_hard: bool

@router.post("/check-video-duplicate", response_model=VideoLinkResponse)
async def check_duplicate(video_link_request: VideoLinkRequest):
    url = video_link_request.url

    try:
        # Асинхронный запрос видео
        response = requests.get(url)
        response.raise_for_status()  # Проверка на успешность запроса
        video_bytes = response.content
    except requests.RequestException as e:
        logger.error(f"Ошибка загрузки видео по ссылке {url}: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid video URL")

    # Подготавливаем данные для Triton
    input_tensors = [grpcclient.InferInput("video_bytes", [1], "BYTES")]
    input_tensors[0].set_data_from_numpy(np.array([video_bytes]))

    try:
        # Инференс через Triton
        results = triton_client.infer(model_name="ensemble_model", inputs=input_tensors)
        classifier_output = results.as_numpy("classifier_output").astype(np.float32)
        video_embed = results.as_numpy("video_output").astype(np.float32)
        audio_embed = results.as_numpy("audio_output").astype(np.float32)

        # Объединение видео и аудио эмбеддингов
        concate_embed = np.concatenate([video_embed, audio_embed], axis=0)


        # Проверка размерности эмбеддинга
        if concate_embed.shape[0] != 1536:  # 1536 (видео) + 1536 (аудио)
            logger.error(f"Неверная размерность эмбеддинга: {concate_embed.shape[0]}")
            raise HTTPException(status_code=500, detail="Invalid embedding size")

        # Проверяем количество эмбеддингов в индексе
        n_embeddings = faiss_index.get_size()
        logger.info(f"Общее количество эмбеддингов: {n_embeddings}")

        if n_embeddings > 0:
            k = min(10, n_embeddings)
            results = faiss_index.search(concate_embed, k)

            is_duplicate, duplicate_id = similarity_model_predict(results, concate_embed)
        else:
            # Если эмбеддингов нет, добавляем новый эмбеддинг в индекс
            is_duplicate = False
            duplicate_id = ""
        faiss_index.add_embedding(concate_embed, "uuid", url)

        return VideoLinkResponse(is_duplicate=is_duplicate, duplicate_for=duplicate_id)

    except Exception as e:
            logger.error(f"Ошибка при инференсе через Triton: {str(e)}")
            raise HTTPException(status_code=500, detail="Error during inference")


@router.post("/process-videos-from-csv/")
async def process_videos_from_csv(file_path: str):
    try:
        # Чтение CSV файла
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Ошибка чтения CSV файла {file_path}: {str(e)}")
        raise HTTPException(status_code=400, detail="Error reading CSV file")

    # Преобразование и сортировка по дате создания


    df['created'] = pd.to_datetime(df['created'])
    df = df.sort_values(by='created')

    # Список для хранения результатов
    ans = []

    for index, row in df.iterrows():
        video_link = row['link']

        try:
            # Запрос видео
            response = requests.get(video_link)
            response.raise_for_status()  # Проверка на успешность запроса
            video_bytes = response.content

            # Подготовка данных для Triton
            input_tensors = [grpcclient.InferInput("video_bytes", [1], "BYTES")]
            input_tensors[0].set_data_from_numpy(np.array([video_bytes]))

            # Инференс через Triton
            results = triton_client.infer(model_name="ensemble_model", inputs=input_tensors)
            classifier_output = results.as_numpy("classifier_output").astype(np.float32)
            video_embed = results.as_numpy("video_output").astype(np.float32)
            audio_embed = results.as_numpy("audio_output").astype(np.float32)

            # Объединение видео и аудио эмбеддингов
            concate_embed = np.concatenate([video_embed, audio_embed], axis=0)


            # Проверка размерности эмбеддинга
            if concate_embed.shape[0] != 1536:  # 1536 (видео) + 1536 (аудио)
                logger.error(f"Неверная размерность эмбеддинга: {concate_embed.shape[0]}")
                raise HTTPException(status_code=500, detail="Invalid embedding size")

            # Проверка количества эмбеддингов в FAISS индексе
            n_embeddings = faiss_index.get_size()

            if n_embeddings > 0:
                # Поиск дубликатов в индексе
                k = 1
                search_results = faiss_index.search(concate_embed, k)


                # Проверка на дубликаты
                is_duplicate, duplicate_id = similarity_model_predict(search_results, concate_embed)
            else:
                # Если эмбеддингов нет, добавляем новый эмбеддинг
                faiss_index.add_embedding(concate_embed, row["uuid"], video_link)
                is_duplicate = False
                duplicate_id = ""

            # Добавление результата в список
            ans.append({
                'created': row['created'].isoformat(),
                'uuid': row['uuid'],
                'link': video_link,
                'is_duplicate': is_duplicate,
                'duplicate_for': duplicate_id
            })

        except Exception as e:
            logger.error(f"Ошибка при обработке видео {video_link}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing video {video_link}, {str(e)}")

        # Создание DataFrame из результатов и сохранение в CSV
        results_df = pd.DataFrame(ans)
        results_df.to_csv('/temp/submission.csv', index=False)

    return {"message": "Videos processed successfully and results saved to submission.csv."}




def infer_model(input_data):
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]



def similarity_model_predict(results, embed):
    distance = np.array([r["distance"] for r in results])
    uuid = [r["uuid"] for r in results]
    embeds_out = [r["embedding"] for r in results]

    combined = np.array(list(zip(distance, uuid, embeds_out)), dtype=object)

    sorted_combined = combined[np.argsort(combined[:, 0])]

    pred_embeds = []

    for distance_sorted, uuid_sorted, embeds_out_sorted in sorted_combined:
        pred_embed = np.concatenate((embed, embeds_out_sorted), axis=0)
        pred_embeds.append(pred_embed)
    similarity_model.eval()
    with torch.no_grad():
        preds = similarity_model(torch.from_array(np.array(pred_embeds, dtype=np.float32)))

    for i, pred_label in enumerate(preds):
        if pred_label > 0.5:
            return True, sorted_combined[i][1]

    return False, ""


