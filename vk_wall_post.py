# =============================================================================
# Модуль для сбора и подготовки текстов постов из ВКонтакте
# =============================================================================
# Данный скрипт выполняет автоматизированный сбор постов из сообществ VK
# для последующего текстового анализа, классификации, тематического моделирования
# и построения аналитики вовлечённости.
#
# Модуль состоит из трёх ключевых компонентов:
#
# 1. get_posts_newsfeed
#    Поиск постов по ключевому слову или фразе с помощью метода `newsfeed.search`.
#    Поддерживает ограничения по времени, постраничную загрузку (через `next_from`)
#    и ограничение общего количества постов (`limit`). Полезен для анализа
#    конкретной темы, события или запроса.
#
# 2. get_posts_wall
#    Получение всех постов на стене сообщества или пользователя через `wall.get`.
#    Позволяет фильтровать публикации по временным границам и содержанию текста.
#    Актуален при построении общей динамики активности или сравнении по годам.
#
# 3. to_dataframe
#    Преобразует «сырой» список словарей, полученных из API, в `pandas.DataFrame`
#    с типизированными и удобно структурированными столбцами:
#    дата, текст, просмотры, лайки, репосты, комментарии, ссылка на пост.
#
# Предварительно в коде происходит авторизация в API VK и задание параметров
# сообщества через `config.py`, откуда импортируются `TOKEN` и `GROUPS`.
#
# Модуль предназначен для интеграции с аналитическим пайплайном:
# - анализа вовлечённости;
# - оценки тональности и настроения контента;
# - тематического моделирования (LDA, BERTopic);
# - сопоставления текстов и фото-аудитории.

import os
import time
import datetime as dt
import pandas as pd
import vk_api
from vk_api.exceptions import ApiError

# Константы проекта (предполагается, что они определяются выше или импортируются)
from config import TOKEN, GROUPS  # убедись, что config.py содержит эти переменные

# Авторизация через токен пользователя/группы
vk = vk_api.VkApi(token=TOKEN)
OWNER_ID = next(iter(GROUPS.values()))

# ----------------------------------------------------------
# Функция поиска постов через newsfeed.search
# ----------------------------------------------------------
def get_posts_newsfeed(query, start_time, end_time, limit=500):
    """
    Получение постов, содержащих определённый текст, через метод newsfeed.search
    Возвращает список словарей с данными о постах
    """
    posts = []
    next_from = ""
    total = 0

    while total < limit:
        try:
            params = {
                "q": query,
                "start_time": start_time,
                "end_time": end_time,
                "count": min(200, limit - total)  # ограничение API
            }
            if next_from:
                params["start_from"] = next_from

            resp = vk.method("newsfeed.search", params)
            items = resp.get("items", [])
            posts.extend(items)
            total += len(items)

            if "next_from" in resp:
                next_from = resp["next_from"]
            else:
                break

            time.sleep(0.34)  # ограничение по скорости запросов API
        except ApiError as e:
            print(f"Ошибка API (newsfeed): {e}")
            break

    return posts

# ----------------------------------------------------------
# Функция поиска постов через wall.get
# ----------------------------------------------------------
def get_posts_wall(owner_id=OWNER_ID, search_text=None, start_time=0, end_time=int(time.time()), limit=500):
    """
    Получение постов с публичной стены сообщества или пользователя
    Фильтрация по дате и ключевому слову в тексте
    """
    posts = []
    offset = 0
    batch = 100  # максимально допустимый размер пакета для wall.get

    while len(posts) < limit:
        try:
            resp = vk.method("wall.get", {
                "owner_id": owner_id,
                "count": batch,
                "offset": offset
            })
        except ApiError as e:
            print(f"Ошибка API (wall): {e}")
            break

        items = resp.get("items", [])
        if not items:
            break

        for post in items:
            date = post["date"]
            if not (start_time <= date <= end_time):
                continue
            if search_text and search_text.lower() not in post.get("text", "").lower():
                continue
            posts.append(post)

        offset += batch

    return posts

# ----------------------------------------------------------
# Преобразование списка постов в pandas DataFrame
# ----------------------------------------------------------
def to_dataframe(posts):
    """
    Преобразует список постов в DataFrame с основными полями: текст, дата, реакции и ссылка
    """
    return pd.DataFrame([{
        'post_id': p['id'],
        'owner_id': p['owner_id'],
        'date': dt.datetime.fromtimestamp(p['date']),
        'year': dt.datetime.fromtimestamp(p['date']).year,
        'month': dt.datetime.fromtimestamp(p['date']).month,
        'text': p.get('text', ''),
        'views': p.get('views', {}).get("count", 0),
        'likes': p.get('likes', {}).get("count", 0),
        'comments': p.get('comments', {}).get("count", 0),
        'reposts': p.get('reposts', {}).get("count", 0),
        'post_url': f"https://vk.com/wall{p['owner_id']}_{p['id']}"
    } for p in posts])
