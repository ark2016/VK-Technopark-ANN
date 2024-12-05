import os
import soundfile as sf
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import string
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.cuda.amp import autocast, GradScaler

# Загрузка данных
# !gdown 1Wd5awSAMCNKlieCv5xnSXy_we7zF9NsG && tar -xf kaggle.tar

# Проверка количества файлов
# !ls -l kaggle/wav | wc -l

# Инициализация модели
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "nyrahealth/CrisperWhisper"
processor = WhisperProcessor.from_pretrained(model_name)
processor.feature_extractor.return_attention_mask = True
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Включение градиентного чекпоинтинга
model.gradient_checkpointing_enable()

# Функция для инференса модели
def batch_inference(model, processor, path_to_wavs, batch_size, sampling_rate=16000):
    results = {}
    wav_files = os.listdir(path_to_wavs)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="russian", task="transcribe")

    scaler = GradScaler()

    for i in tqdm(range(0, len(wav_files), batch_size), total=np.ceil(len(wav_files) / batch_size)):
        audio_paths = wav_files[i : i + batch_size]

        batch = []

        for path in audio_paths:
            audio, _ = sf.read(os.path.join(path_to_wavs, path))
            batch.append(audio)
        inputs = processor(batch, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

        x, x_masks = inputs["input_features"].to(device), inputs["attention_mask"].to(device)

        with torch.no_grad():
            with autocast():
                model.eval()
                output_ids = model.generate(x, forced_decoder_ids=forced_decoder_ids, attention_mask=x_masks)

        transcribtion = processor.batch_decode(output_ids, skip_special_tokens=True)

        results.update(zip(audio_paths, transcribtion))

        # Очистка кэша CUDA
        torch.cuda.empty_cache()

    return results

# Параметры для инференса
path_to_wavs = "kaggle/wav"
batch_size = 1  # Уменьшаем размер пакета

# Выполнение инференса
results = batch_inference(model=model, processor=processor, path_to_wavs=path_to_wavs, batch_size=batch_size)

# Постпроцессинг данных
def dummy_postprocessing(data):
    for filename, hypo in data.items():
        hypo = hypo.strip()
        hypo = hypo.translate(str.maketrans('', '', string.punctuation))
        hypo = hypo.lower()
        data[filename] = hypo
    return data

clean_data = dummy_postprocessing(results)

# Подготовка файла для загрузки на Kaggle
sample_submission = pd.read_csv("kaggle/sample_submission.csv")
sample_submission["hypo"] = sample_submission["filename"].apply(lambda x: clean_data[x])
sample_submission.insert(0, "id", sample_submission.index)

# Замена пустых строк на строки с пробелом
sample_submission["hypo"] = sample_submission["hypo"].replace('', ' ')

# Сохранение файла
sample_submission.to_csv("CrisperWhisper.csv", index=False)

