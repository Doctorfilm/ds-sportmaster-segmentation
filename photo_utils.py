# =============================================================================
# Модуль обработки фотографий VK-сообществ: загрузка, детекция лиц, анализ
# =============================================================================
# Данный блок кода реализует полный пайплайн анализа фотографий с мероприятий,
# опубликованных в альбомах сообществ ВКонтакте. Основная цель — автоматическое
# построение демографического и эмоционального портрета аудитории на основе
# изображений с мероприятий.
#
# Весь процесс включает в себя следующие этапы:
#
# 1. Загрузка фотографий (`vk_download_photos`)
#    Фотографии выгружаются из указанных альбомов сообщества по API VK.
#    Поддерживается автоматическое определение года по дате создания альбома,
#    выбор наилучшего доступного размера изображения, сохранение с именованием
#    по шаблону <группа>_<год>_<номер>.jpg.
#
# 2. Детекция и сохранение лиц (`vk_detect_and_save_faces`)
#    С помощью DeepFace и заданной модели (например, VGG-Face) из всех загруженных
#    изображений выделяются фрагменты с лицами. Для повышения качества:
#      - отбрасываются слишком размытые участки (Laplacian Variance),
#      - применяется порог по confidence и минимальный размер лица.
#    Каждый валидный фрагмент сохраняется как отдельное изображение,
#    одновременно рассчитывается embedding для последующей фильтрации.
#
# 3. Фильтрация повторяющихся лиц и анализ выражений (`process_all_faces`)
#    Все найденные лица группируются по embedding-похожести. Дублирующиеся
#    фрагменты отбрасываются, остаются только уникальные. Для каждого лица
#    проводится анализ возраста, пола и эмоции. Результаты сохраняются в
#    итоговый CSV-файл, пригодный для дальнейшей визуализации и статистики.
#
# Используемые библиотеки:
# - `vk_api` — для загрузки изображений с VK;
# - `DeepFace` — детекция лиц, вычисление эмбеддингов и эмоциональный анализ;
# - `OpenCV` и `Pillow` — для чтения, обработки и сохранения изображений;
# - `pandas` и `numpy` — для работы с табличными и числовыми данными.
#
# Конечный результат — структурированный CSV-файл с уникальными лицами и
# соответствующими признаками: возраст, пол, доминирующая эмоция, дата анализа.


import os
import requests
import shutil
import pickle
import cv2
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import vk_api
from deepface import DeepFace
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# -------- Загрузка фото из VK -----------------------------
# ----------------------------------------------------------

def resolve_group_id(vk, group_identifier):
    """
    Конвертирует короткое имя сообщества (screen_name) в числовой ID
    Если передан числовой идентификатор, возвращает его без изменений
    """
    if isinstance(group_identifier, int):
        return group_identifier
    try:
        result = vk.method("utils.resolveScreenName", {"screen_name": group_identifier})
        if result and result["type"] == "group":
            return -result["object_id"]
        raise ValueError(f"Группа {group_identifier} не найдена")
    except Exception as e:
        print(f"Ошибка при получении ID: {e}")
        return None

def vk_download_photos(P_TOKEN, P_GROUPS, P_ALBUM_IDS, P_PHOTO_PATH, P_MAX_PHOTOS):
    """
    Скачивает фотографии из указанных альбомов VK-сообществ и сохраняет по годам
    """
    vk = vk_api.VkApi(token=P_TOKEN)

    for group_name, group_id in P_GROUPS.items():
        resolved_id = resolve_group_id(vk, group_id)
        if not resolved_id:
            continue

        album_ids = P_ALBUM_IDS.get(group_name)
        if not album_ids:
            print(f"Пропущена группа {group_name}: album_ids не заданы.")
            continue

        try:
            albums = vk.method("photos.getAlbums", {
                "owner_id": resolved_id,
                "need_system": 1
            })
        except Exception as e:
            print(f"Не удалось получить альбомы группы {group_name}: {e}")
            continue

        album_years = {}
        for album in albums.get("items", []):
            if album["id"] in album_ids:
                dt = datetime.fromtimestamp(album["created"])
                album_years[album["id"]] = str(dt.year)

        for album_id in album_ids:
            year = album_years.get(album_id, str(album_id))
            save_dir = os.path.join(P_PHOTO_PATH, year)

            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)

            print(f"\nЗагрузка фотографий: группа {group_name}, альбом {album_id}, папка {year}")

            downloaded = 0

            try:
                photos = vk.method("photos.get", {
                    "owner_id": resolved_id,
                    "album_id": album_id,
                    "extended": 0,
                    "count": 1000
                })

                sorted_photos = sorted(photos.get("items", []), key=lambda x: x["date"], reverse=True)

                for photo in tqdm(sorted_photos, desc=f"{group_name}_{year}"):
                    if downloaded >= P_MAX_PHOTOS:
                        break
                    try:
                        url = None
                        for ptype in ["z", "y", "x", "m", "s"]:
                            url = next((size["url"] for size in photo["sizes"] if size["type"] == ptype), None)
                            if url:
                                break
                        if not url and "sizes" in photo:
                            url = photo["sizes"][0]["url"]

                        response = requests.get(url, timeout=10)
                        img = Image.open(BytesIO(response.content))
                        filename = f"{group_name}_{year}_{downloaded + 1:04d}.jpg"
                        img.save(os.path.join(save_dir, filename))
                        downloaded += 1
                    except Exception:
                        pass

            except Exception:
                pass

            print(f"Загружено {downloaded} фото в {save_dir}")

# ----------------------------------------------------------
# -------- Обработка лиц через DeepFace --------------------
# ----------------------------------------------------------

MIN_CONFIDENCE = 0.90
BLUR_THRESHOLD = 200.0
FACE_SIZE_MIN = 30
MODEL_NAME = "VGG-Face"
DETECT_BACK = "retinaface"

def is_blurry(image, threshold=BLUR_THRESHOLD):
    """Оценка чёткости изображения по дисперсии Лапласиана"""
    img = np.clip(image, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def vk_detect_and_save_faces(P_PHOTO_PATH, P_FACE_PATH):
    """
    Обнаруживает лица на фото и сохраняет их отдельно с эмбеддингами
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = DeepFace.build_model(MODEL_NAME)

    for year_folder in sorted(os.listdir(P_PHOTO_PATH)):
        src = os.path.join(P_PHOTO_PATH, year_folder)
        if not os.path.isdir(src):
            continue
        print(f"\n=== Поиск лиц: {year_folder} ===")

        dst = os.path.join(P_FACE_PATH, year_folder)
        embeddings_path = os.path.join(P_FACE_PATH, f"{year_folder}_embeddings.pkl")
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst, exist_ok=True)

        embeddings = []
        if os.path.exists(embeddings_path):
            os.remove(embeddings_path)

        counter = 0
        files = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        for fname in tqdm(files, desc=f"Лица {year_folder}"):
            img_path = os.path.join(src, fname)
            full_img = cv2.imread(img_path)
            if full_img is None:
                continue
            try:
                detections = DeepFace.extract_faces(
                    img_path=img_path,
                    detector_backend=DETECT_BACK,
                    enforce_detection=False,
                    align=False
                )
                for face_obj in detections:
                    confidence = face_obj.get('confidence', 1.0)
                    if confidence < MIN_CONFIDENCE:
                        continue
                    area = face_obj.get('facial_area')
                    if not area:
                        continue
                    x, y, w, h = area['x'], area['y'], area['w'], area['h']
                    if w < FACE_SIZE_MIN or h < FACE_SIZE_MIN:
                        continue
                    face_crop = full_img[y:y+h, x:x+w]
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    if is_blurry(face_rgb):
                        continue

                    save_name = f"{year_folder}_{counter:05d}.jpg"
                    save_path = os.path.join(dst, save_name)
                    cv2.imwrite(save_path, cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
                    emb = DeepFace.represent(
                        img_path=save_path,
                        model_name=MODEL_NAME,
                        enforce_detection=False,
                        detector_backend='skip'
                    )[0]['embedding']
                    embeddings.append((emb, save_name, fname))
                    counter += 1
            except Exception as e:
                print(f"Ошибка {fname}: {e}")

        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Сохранено {counter} лиц в {dst}")

# ----------------------------------------------------------
# -------- Фильтрация повторяющихся лиц --------------------
# ----------------------------------------------------------

def process_all_faces(P_FACE_PATH, P_UNIQUE_PATH, P_OUTPUT_CSV, P_THRESHOLD):
    """
    Удаляет дублирующиеся лица и сохраняет уникальные с анализом пола, возраста, эмоций
    """
    emotion_order = ['fair', 'angry', 'sad', 'neutral', 'happy', 'surprise']
    rows = []

    DeepFace.build_model("VGG-Face")
    detector_backend = "skip"

    for fname in sorted(os.listdir(P_FACE_PATH)):
        if not fname.endswith("_embeddings.pkl"):
            continue
        year = fname.replace("_embeddings.pkl", "")
        pkl_path = os.path.join(P_FACE_PATH, fname)

        try:
            with open(pkl_path, "rb") as f:
                embeddings_info = pickle.load(f)
            emb_array = np.stack([e for e, _, _ in embeddings_info])
        except:
            continue

        used = np.zeros(len(emb_array), bool)
        unique_idx = []
        for i in range(len(emb_array)):
            if used[i]:
                continue
            unique_idx.append(i)
            if i + 1 >= len(emb_array):
                continue
            sims = cosine_similarity(emb_array[i:i+1], emb_array[i+1:])[0]
            dup = np.where(1 - sims < P_THRESHOLD)[0] + (i + 1)
            used[dup] = True

        out_dir = os.path.join(P_UNIQUE_PATH, year)
        os.makedirs(out_dir, exist_ok=True)

        for face_number, idx in enumerate(tqdm(unique_idx, desc=f"Анализ {year}")):
            _, face_img, _ = embeddings_info[idx]
            filename = f"{year}_face_{face_number:05d}.jpg"
            save_path = os.path.join(out_dir, filename)
            try:
                if isinstance(face_img, str):
                    face_img = cv2.imread(os.path.join(P_FACE_PATH, year, face_img))
                    if face_img is None:
                        continue
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                elif not isinstance(face_img, np.ndarray):
                    face_img = np.array(face_img)
                if face_img.dtype != np.uint8:
                    face_img = (face_img * 255).astype(np.uint8) if face_img.max() <= 1.0 else face_img.astype(np.uint8)
                bgr_face = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr_face)
            except:
                continue

            try:
                res = DeepFace.analyze(
                    img_path=save_path,
                    actions=["age", "gender", "emotion"],
                    enforce_detection=False,
                    detector_backend=detector_backend,
                    align=False,
                    silent=True
                )
                face = res[0] if isinstance(res, list) else res
                g = face["gender"]
                gender = "Woman" if g["Woman"] > g["Man"] else "Man"
                emo = face["emotion"]
                emotion = max(emo, key=emo.get)
                rows.append({
                    "filename": filename,
                    "face_number": face_number + 1,
                    "gender": gender,
                    "gender_score": g[gender],
                    "emotion": emotion,
                    "emotion_score": emo[emotion],
                    "age": face["age"],
                    "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "folder": year
                })
            except:
                continue

    if not rows:
        return

    df = pd.DataFrame(rows)
    df["emotion"] = pd.Categorical(df["emotion"], categories=emotion_order, ordered=True)
    df["emotion_rank"] = df["emotion"].cat.codes + 1
    df.to_csv(P_OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Сохранено итоговое CSV: {P_OUTPUT_CSV}")
