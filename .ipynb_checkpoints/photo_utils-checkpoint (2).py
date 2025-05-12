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

# ----------------------------------------------------------
# -------- Загрузка фото из VK -----------------------------
# ----------------------------------------------------------

def resolve_group_id(vk, group_identifier):
    """
    Получает owner_id группы по её screen_name или возвращает числовой ID.
    """
    if isinstance(group_identifier, int):
        return group_identifier
    try:
        result = vk.method("utils.resolveScreenName", {"screen_name": group_identifier})
        if result and result.get("type") == "group":
            return -result.get("object_id")
        raise ValueError(f"Группа {group_identifier} не найдена")
    except Exception as e:
        print(f"Ошибка при получении ID: {e}")
        return None


def download_photos(token, groups, albums,
                    photo_path='data/raw',
                    max_photos=1000,
                    pref_sizes=None):
    """
    Загружает фотографии из указанных VK-групп и альбомов.
    - token: VK API token
    - groups: dict {group_name: screen_name_or_id}
    - albums: dict {group_name: [album_id, ...]}
    - photo_path: папка для сохранения фото (будут созданы вложенные папки по годам)
    - max_photos: макс. число фото на альбом
    - pref_sizes: список приоритетных типов размеров, напр. ['z','y','x']
    """
    max_photos = int(max_photos)
    if pref_sizes is None:
        pref_sizes = ["z", "y", "x", "m", "s"]

    vk = vk_api.VkApi(token=token)

    for group_name, group_id in groups.items():
        owner_id = resolve_group_id(vk, group_id)
        if not owner_id:
            continue

        album_ids = albums.get(group_name, [])
        if not album_ids:
            print(f"Пропущена группа {group_name}: album_ids не заданы.")
            continue

        try:
            resp = vk.method("photos.getAlbums", {"owner_id": owner_id, "need_system": 1})
        except Exception as e:
            print(f"Не удалось получить альбомы группы {group_name}: {e}")
            continue

        # Сопоставление album_id -> год создания
        album_years = {
            alb.get("id"): str(datetime.fromtimestamp(alb.get("created")).year)
            for alb in resp.get("items", []) if alb.get("id") in album_ids
        }

        for album_id in album_ids:
            year = album_years.get(album_id, str(album_id))
            save_dir = os.path.join(photo_path, year)
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)

            print(f"\n📥 Загрузка фото: группа={group_name}, альбом={album_id}, папка={year}")
            downloaded = 0

            try:
                photos = vk.method("photos.get", {
                    "owner_id": owner_id,
                    "album_id": album_id,
                    "extended": 0,
                    "count": 1000
                })
                items = sorted(photos.get("items", []), key=lambda x: x.get("date", 0), reverse=True)

                for photo in tqdm(items, desc=f"{group_name}_{year}"):
                    if downloaded >= max_photos:
                        break
                    # Выбор URL приоритетного размера
                    url = None
                    for sz in photo.get("sizes", []):
                        if sz.get("type") in pref_sizes:
                            url = sz.get("url")
                            break
                    if not url and photo.get("sizes"):
                        url = photo["sizes"][0].get("url")

                    try:
                        resp = requests.get(url, timeout=10)
                        img = Image.open(BytesIO(resp.content))
                        filename = f"{group_name}_{year}_{downloaded+1:04d}.jpg"
                        img.save(os.path.join(save_dir, filename))
                        downloaded += 1
                    except Exception as e:
                        print(f"Ошибка загрузки фото URL={url}: {e}")

            except Exception as e:
                print(f"Ошибка при получении фото из альбома {album_id}: {e}")

            print(f"Загружено {downloaded} фото в {save_dir}")


# ----------------------------------------------------------
# -------- Обработка лиц через DeepFace --------------------
# ----------------------------------------------------------

def is_blurry(image, threshold=100.0):
    """
    Проверка размытости по дисперсии Лапласиана.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def simple_preprocess_face(img, target_size=(224, 224)):
    """
    Минимальная предобработка лица для DeepFace: RGB->BGR, resize и нормировка.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def detect_and_save_faces(photo_path, face_path, face_size=50,
                          model_name="VGG-Face", detector_backend="retinaface"):
    """
    Ищет лица в папках по годам, проверяет размер и размытость,
    сохраняет фрагменты и эмбеддинги.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = DeepFace.build_model(model_name)

    for year in sorted(os.listdir(photo_path)):
        src = os.path.join(photo_path, year)
        if not os.path.isdir(src):
            continue
        print(f"\n=== Обработка лиц: {year} ===")

        dst_folder = os.path.join(face_path, year)
        os.makedirs(dst_folder, exist_ok=True)
        embeddings_info = []
        counter = 0

        files = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"Найдено {len(files)} изображений...")

        for fname in tqdm(files, desc=f"Лица {year}"):
            img_file = os.path.join(src, fname)
            try:
                detections = DeepFace.extract_faces(
                    img_path=img_file,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                    align=False
                )
                for face_obj in detections:
                    face_img = face_obj.get("face")
                    face_img = np.nan_to_num(face_img)
                    # Приведение к uint8
                    if face_img.dtype != np.uint8:
                        face_img = np.clip(face_img * 255, 0, 255).astype(np.uint8)

                    h, w = face_img.shape[:2]
                    if min(h, w) < face_size or is_blurry(face_img):
                        continue

                    save_name = f"{year}_{counter:05d}.jpg"
                    save_path = os.path.join(dst_folder, save_name)
                    cv2.imwrite(save_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

                    emb = DeepFace.represent(
                        img_path=save_path,
                        model_name=model_name,
                        enforce_detection=False,
                        detector_backend="skip"
                    )[0]["embedding"]
                    embeddings_info.append((emb, save_name, fname))
                    counter += 1
            except Exception as e:
                print(f"Ошибка обработки {fname}: {e}")

        # Сохраняем эмбеддинги
        out_file = os.path.join(face_path, f"{year}_embeddings.pkl")
        with open(out_file, "wb") as f:
            pickle.dump(embeddings_info, f)
        print(f"Сохранено {counter} лиц в {dst_folder}")
        print(f"Эмбеддинги в {out_file}")