# =============================================================================
# Блок визуализации распределений: гистограммы, barplot, KDE и тренды
# =============================================================================
# Данный блок кода содержит функции для базовой визуальной аналитики данных.
# Он разработан с целью быстрой и наглядной диагностики распределений признаков
# в DataFrame, а также сравнения этих распределений по дополнительным категориям
# (например, по годам, полу, уровням вовлечённости и т.п.).
#
# Каждая функция принимает:
#   - входной DataFrame (`df`);
#   - список столбцов для анализа;
#   - список имён (заголовков графиков);
#   - необязательный параметр `hue` для группировки по категориям.
#
# Описание функций:
# - hist_graph: отображает гистограммы распределения признаков по группам;
# - bar_graph: строит barplot-графики (например, динамику лайков/просмотров по годам);
# - kde_graph_count: визуализирует агрегированные величины (например, средние значения по годам);
# - kde_graph: строит сглаженные графики плотности (KDE) для числовых признаков,
#              и линейные графики пропорций — для категориальных.
#
# Все графики оформлены в едином стиле и адаптированы для быстрой вставки в отчёт
# или дашборд. Функции поддерживают параметр `suptitle` — для заголовка всей фигуры.

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Устанавливаем стиль графиков
sns.set_style("darkgrid")

# =============================================================================
# Функция для построения гистограмм распределения признаков
# =============================================================================
def hist_graph(df, list_pl, list_pl_name, suptitle='', p_hue=None):
    max_row = 2
    plt.figure(figsize=[15, 4 * max_row])
    plt.suptitle(suptitle, ha='center', fontsize=25)
    plt.subplots_adjust(top=0.93, hspace=0.3)

    for i, param, name in zip(range(len(list_pl)), list_pl, list_pl_name):
        plt.subplot(max_row, 2, i + 1)
        sns.histplot(
            data=df,
            x=param,
            hue=p_hue,
            bins=10,
            multiple='dodge',
            shrink=0.8
        )
        plt.xlabel(name, fontsize=10)
        plt.ylabel('Частота', fontsize=10)
        plt.title(name, fontsize=15)

    plt.show()

# =============================================================================
# Функция для построения barplot-графиков по нескольким признакам
# =============================================================================
def bar_graph(df, x_name, list_pl, list_pl_name, suptitle='', p_hue=None):
    max_row = 2
    plt.figure(figsize=[15, 5 * max_row])
    plt.suptitle(suptitle, ha='center', fontsize=20)
    plt.subplots_adjust(top=0.87, hspace=0.3)

    for i, param, name in zip(range(len(list_pl)), list_pl, list_pl_name):
        plt.subplot(max_row, 2, i + 1)
        sns.barplot(
            data=df,
            x=x_name,
            y=param,
            hue=p_hue
        )
        plt.xlabel('Год', fontsize=10)
        plt.ylabel(name, fontsize=10)
        plt.title(name, fontsize=12)

    plt.show()

# =============================================================================
# Функция для построения линейных графиков агрегированных количеств
# =============================================================================
def kde_graph_count(df, hue_val, agg_kwargs, titles, suptitle=''):
    # Агрегируем данные по заданному признаку (обычно это 'year')
    annual = (
        df
        .groupby(hue_val)
        .agg(**agg_kwargs)
        .sort_index()
    )

    # Настройка фигуры и палитры
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    rows, cols = 2, 2
    plt.figure(figsize=(15, 4 * rows))
    plt.suptitle(suptitle, fontsize=20)
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)

    # Отрисовка каждого показателя на своей оси
    for idx, param in enumerate(agg_kwargs):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.plot(
            annual.index,
            annual[param],
            marker='o',
            color=palette[idx]
        )
        ax.set_title(titles[param], fontsize=14)
        ax.set_xlabel(hue_val.capitalize())
        ax.set_ylabel(titles[param])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()

# =============================================================================
# Функция для построения графиков относительного распределения признаков
# Строит:
# - KDE-графики для числовых признаков;
# - Линейные графики пропорций для категориальных.
# =============================================================================
def kde_graph(df, list_pl, list_pl_name, suptitle='', p_hue=None):

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    rows = 2
    plt.figure(figsize=(15, 4 * rows))
    plt.suptitle(suptitle, ha='center', fontsize=20)
    plt.subplots_adjust(top=0.90, hspace=0.3)

    for idx, (param, name) in enumerate(zip(list_pl, list_pl_name)):
        ax = plt.subplot(rows, 2, idx + 1)

        if pd.api.types.is_numeric_dtype(df[param]):
            # Построение KDE-графика
            sns.kdeplot(
                data=df,
                x=param,
                hue=p_hue,
                palette=palette,
                common_norm=False,
                fill=False,
                ax=ax
            )
            ax.set_ylabel('Плотность')
        else:
            # Построение пропорций для категориального признака
            prop_df = (
                df
                .groupby([param, p_hue])
                .size()
                .unstack(fill_value=0)
                .apply(lambda col: col / col.sum(), axis=0)
            )
            for i, hue_val in enumerate(prop_df.columns):
                ax.plot(
                    prop_df.index.astype(str),
                    prop_df[hue_val],
                    marker='o',
                    color=palette[i],
                    label=str(hue_val)
                )
            ax.set_ylabel('Доля')

        ax.set_xlabel(name)
        ax.set_title(name)

    plt.show()
