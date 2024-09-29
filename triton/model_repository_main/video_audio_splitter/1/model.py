import tempfile
import cv2
from PIL import Image
from transformers import AutoFeatureExtractor, Wav2Vec2Processor
import triton_python_backend_utils as pb_utils
import librosa
import ffmpeg
import soundfile as sf
import io
import numpy as np

class TritonPythonModel:

    def initialize(self, args=None):
        self.video_feature_extractor = AutoFeatureExtractor.from_pretrained("/workspace/video_feature_extractor")
        print("well done")
        self.audio_feature_extractor = Wav2Vec2Processor.from_pretrained("/workspace/audio_feature_extractor")



    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "video_bytes")

            y, video = self.load_video(input_tensor.as_numpy()[0])
            postprocessing_video = self.video_feature_extractor(images=video, return_tensors="np")

            print(y.shape)


            audio_batches = self.audio_feature_extractor([y], return_tensors="np", padding="longest").input_values

            # Создаем выходной тензор
            output_tensor = pb_utils.Tensor("video_features", postprocessing_video['pixel_values'])

            audio_tensor = pb_utils.Tensor("audio_frames", audio_batches)
            responses.append(pb_utils.InferenceResponse([output_tensor, audio_tensor]))


        return responses

    def load_video(self, video_bytes, count=4, target_size=(224, 224)):
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(video_bytes)

            cap = cv2.VideoCapture(temp.name)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = frame_count // count
            frames = []

            try:

                audio_data, sample_rate = self.extract_audio_from_video(temp.name)

                y = librosa.resample(audio_data.T, orig_sr=sample_rate,
                                     target_sr=16000)

                print(y.shape)

                if y.ndim > 1:
                    y = np.mean(y, axis=0)
                    print(y.shape)
            except:
                y = np.zeros((768))

            for frame_index in range(0, count):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index * step)
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, target_size)
                frames.append(Image.fromarray(frame))

            cap.release()
            return y, frames

    def extract_audio_from_video(self, video_path, target_sample_rate=16000):
        # Используем ffmpeg для извлечения аудио в виде сырого PCM потока
        out, _ = (
            ffmpeg.input(video_path)
            .output('pipe:', format='wav', acodec='pcm_s16le', ar=target_sample_rate)
            .run(capture_stdout=True, capture_stderr=True)
        )

        audio_bytes = io.BytesIO(out)

        audio_data, sr = sf.read(audio_bytes)

        return audio_data, sr
