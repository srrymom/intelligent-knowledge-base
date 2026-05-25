"""  # Начало многострочного описания назначения скрипта.
Одноразовый скрипт для подготовки бенчмарк-данных.  # Скрипт нужен для разовой загрузки тестовых данных.
Скачивает 50 аудиофайлов из датасета Golos (sberdevices_golos_10h_crowd).  # Источник данных и размер выборки.
Сохраняет WAV + samples.jsonl в data/benchmark/asr/.  # Итоговые аудиофайлы и манифест кладутся в эту папку.

Запустить один раз перед eval_asr.py.  # Данные нужно подготовить до запуска оценки ASR.
Нужен env с datasets и soundfile.  # В окружении должны быть установлены нужные библиотеки.
"""  # Конец многострочного описания.

import io  # Даёт BytesIO, чтобы читать аудио из байтов как из файла в памяти.
import json  # Нужен для записи метаданных в формате JSON Lines.
import os  # Нужен для создания папок на диске.
import numpy as np  # Используется для приведения аудиомассива к float32 перед сохранением.
import soundfile as sf  # Читает и записывает аудиофайлы.
from datasets import load_dataset, Audio  # Загружает датасет и задаёт режим работы с аудиоколонкой.

# Загружаем без автодекодирования аудио — datasets не будет дёргать torchcodec.
ds = load_dataset(  # Создаём потоковый объект датасета Hugging Face.
    "bond005/sberdevices_golos_10h_crowd",  # Имя датасета Golos на Hugging Face.
    split="test",  # Берём тестовую часть датасета.
    streaming=True,  # Читаем данные потоком, не скачивая весь датасет целиком.
    trust_remote_code=True,  # Разрешаем выполнить код датасета, если он нужен загрузчику.
)  # Завершаем вызов load_dataset.
ds = ds.cast_column("audio", Audio(decode=False))  # Оставляем аудио как сырые байты без автоматического декодирования.

os.makedirs("data/benchmark/asr/audio", exist_ok=True)  # Создаём папку для WAV-файлов, если её ещё нет.

manifest = []  # Здесь будет список записей для будущего samples.jsonl.
for i, s in enumerate(ds):  # Идём по датасету, получая индекс i и пример s.
    if i >= 50:  # Останавливаемся после первых 50 аудиопримеров.
        break  # Выходим из цикла загрузки.

    audio_bytes = s["audio"]["bytes"]  # Достаём сырые байты аудиофайла из текущего примера.
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))  # Декодируем байты в массив звука и частоту дискретизации.
    if audio_array.ndim > 1:  # Проверяем, есть ли у аудио несколько каналов.
        audio_array = audio_array.mean(axis=1)  # Преобразуем стерео или многоканальный звук в моно.

    path = f"data/benchmark/asr/audio/{i:04d}.wav"  # Формируем путь вида 0000.wav, 0001.wav и так далее.
    sf.write(path, audio_array.astype(np.float32), sr)  # Сохраняем аудио в WAV в формате float32.
    manifest.append({"id": i, "audio": path, "transcription": s["transcription"]})  # Добавляем запись о файле и тексте.
    print(f"  [{i + 1}/50] {path}")  # Печатаем прогресс подготовки данных.

with open("data/benchmark/asr/samples.jsonl", "w", encoding="utf-8") as f:  # Открываем манифест для записи в UTF-8.
    for m in manifest:  # Проходим по всем подготовленным записям манифеста.
        f.write(json.dumps(m, ensure_ascii=False) + "\n")  # Записываем одну JSON-запись на строку.

print(f"\nГотово: {len(manifest)} файлов → data/benchmark/asr/samples.jsonl")  # Сообщаем итоговое число файлов и путь.
