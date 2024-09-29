from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .utils import VideoDuplicateSearcherCosine
import numpy as np
import requests
import tritonclient.grpc as grpcclient
import pandas as pd
import logging
import onnxruntime as ort

# Создание маршрутизатора для FastAPI
router = APIRouter()

# Инициализация сессии ONNX для модели
ort_session = ort.InferenceSession("./weights/similarity_model11.onnx")

# Инициализация Triton клиента и FAISS индекса
triton_client = grpcclient.InferenceServerClient(url="triton:8001")
faiss_index = VideoDuplicateSearcherCosine(1536)  # Размерность 1536 (видео) + 1536 (аудио)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoLinkRequest(BaseModel):
    """
    Модель для запроса с видео URL.
    """
    url: str

class VideoLinkResponse(BaseModel):
    """
    Модель для ответа с информацией о дубликате.
    """
    is_duplicate: bool
    duplicate_for: str

class VideoData(BaseModel):
    """
    Модель для хранения данных о видео.
    """
    created: str
    uuid: str
    link: str
    is_duplicate: bool
    duplicate_for: str
    is_hard: bool

@router.post("/check-video-duplicate", response_model=VideoLinkResponse)
async def check_duplicate(video_link_request: VideoLinkRequest):
    """
    Проверяет, является ли видео дубликатом по предоставленной ссылке.

    Args:
        video_link_request (VideoLinkRequest): Запрос, содержащий URL видео.

    Returns:
        VideoLinkResponse: Ответ с информацией о том, является ли видео дубликатом и его идентификатором.
    """
    url = video_link_request.url

    try:
        response = requests.get(url)
        response.raise_for_status()
        video_bytes = response.content
    except requests.RequestException as e:
        logger.error(f"Ошибка загрузки видео по ссылке {url}: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid video URL")

    input_tensors = [grpcclient.InferInput("video_bytes", [1], "BYTES")]
    input_tensors[0].set_data_from_numpy(np.array([video_bytes]))

    try:
        results = triton_client.infer(model_name="ensemble_model", inputs=input_tensors)
        video_embed = results.as_numpy("video_output").astype(np.float32)
        audio_embed = results.as_numpy("audio_output").astype(np.float32)

        concate_embed = np.concatenate([video_embed, audio_embed], axis=0)

        if concate_embed.shape[0] != 1536:
            logger.error(f"Неверная размерность эмбеддинга: {concate_embed.shape[0]}")
            raise HTTPException(status_code=500, detail="Invalid embedding size")

        n_embeddings = faiss_index.get_size()
        logger.info(f"Общее количество эмбеддингов: {n_embeddings}")

        if n_embeddings > 0:
            k = min(10, n_embeddings)
            results = faiss_index.search(concate_embed, k)
            is_duplicate, duplicate_id = similarity_model_predict(results, concate_embed)
        else:
            is_duplicate = False
            duplicate_id = ""

        faiss_index.add_embedding(concate_embed, url[42:].split(".")[0], url)
        return VideoLinkResponse(is_duplicate=is_duplicate, duplicate_for=duplicate_id)

    except Exception as e:
        logger.error(f"Ошибка при инференсе через Triton: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during inference")

@router.post("/process-videos-from-csv/")
async def process_videos_from_csv(file_path: str):
    """
    Обрабатывает видео из CSV файла, проверяя их на дубликаты.

    Args:
        file_path (str): Путь к CSV файлу, содержащему ссылки на видео.

    Returns:
        dict: Сообщение об успешной обработке и сохранении результатов.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Ошибка чтения CSV файла {file_path}: {str(e)}")
        raise HTTPException(status_code=400, detail="Error reading CSV file")

    df['created'] = pd.to_datetime(df['created'])
    df = df.sort_values(by='created')

    results = []

    for index, row in df.iterrows():
        video_link = row['link']

        try:
            response = requests.get(video_link)
            response.raise_for_status()
            video_bytes = response.content

            input_tensors = [grpcclient.InferInput("video_bytes", [1], "BYTES")]
            input_tensors[0].set_data_from_numpy(np.array([video_bytes]))

            results = triton_client.infer(model_name="ensemble_model", inputs=input_tensors)
            video_embed = results.as_numpy("video_output").astype(np.float32)
            audio_embed = results.as_numpy("audio_output").astype(np.float32)

            concate_embed = np.concatenate([video_embed, audio_embed], axis=0)

            if concate_embed.shape[0] != 1536:
                logger.error(f"Неверная размерность эмбеддинга: {concate_embed.shape[0]}")
                raise HTTPException(status_code=500, detail="Invalid embedding size")

            n_embeddings = faiss_index.get_size()

            if n_embeddings > 0:
                search_results = faiss_index.search(concate_embed, 1)
                is_duplicate, duplicate_id = similarity_model_predict(search_results, concate_embed)
            else:
                is_duplicate = False
                duplicate_id = ""

            faiss_index.add_embedding(concate_embed, row["uuid"], video_link)

            results.append({
                'created': row['created'].isoformat(),
                'uuid': row['uuid'],
                'link': video_link,
                'is_duplicate': is_duplicate,
                'duplicate_for': duplicate_id
            })

        except Exception as e:
            logger.error(f"Ошибка при обработке видео {video_link}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing video {video_link}, {str(e)}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('/temp/submission.csv', index=False)

    return {"message": "Videos processed successfully and results saved to submission.csv."}

def infer_model(input_data):
    """
    Выполняет инференс модели ONNX.

    Args:
        input_data (np.ndarray): Входные данные для модели.

    Returns:
        np.ndarray: Результаты инференса.
    """
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

def similarity_model_predict(results, embed):
    """
    Определяет, является ли видео дубликатом, на основе результатов поиска и дополнительного инференса.

    Args:
        results (list): Результаты поиска в FAISS индексе.
        embed (np.ndarray): Эмбеддинг видео.

    Returns:
        tuple: Булево значение, указывающее на дубликат, и идентификатор дубликата.
    """
    distance = np.array([r["distance"] for r in results])
    uuid = [r["uuid"] for r in results]
    embeds_out = [r["embed"] for r in results]

    combined = np.array(list(zip(distance, uuid, embeds_out)), dtype=object)
    sorted_combined = combined[np.argsort(combined[:, 0])[::-1]]

    if sorted_combined[0][0] > 0.75:
        return True, sorted_combined[0][1]

    pred_embeds = [
        np.concatenate((embeds_out_sorted, embed), axis=0)
        for _, _, embeds_out_sorted in sorted_combined
    ]

    preds = infer_model(np.array(pred_embeds, dtype=np.float32))
    max_pred = max(preds)
    max_index = np.argmax(preds)

    if max_pred > 0.5:
        return True, sorted_combined[max_index][1]
    return False, ""