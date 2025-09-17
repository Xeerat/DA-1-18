import matplotlib.pyplot as plt
# load_boston() устарел и больше не используется
from sklearn.datasets import fetch_california_housing


def detect_emission(data, coef=1.5):
    """
    Поиск выбросов по методу IQR.

    Входные данные:
        data (pandas.Series) -  ряд чисел
        coef (float) - Коэффициент для IQR. По умолчанию 1.5.

    Возвращает:
        (pandas.Series) - ряд из True/False, где True - это выброс
    """
    # Определение границ
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - coef * IQR
    upper = Q3 + coef * IQR

    # Числа вне границ являются выбросами
    return (data < lower) | (data > upper)


def create_graph(data, emissions):
    """
    Создает график с выделением выбросов

    Входные данные :
        data (pandas.Series) - ряд чисел, по которым строиться график.
        emissions (pandas.Series) - ряд из True/False, где True — выброс.

    Возвращает:
        Ничего, только строит график
    """
    plt.figure(figsize=(16, 8))
    plt.scatter(range(len(data[emissions])), data[emissions], color="red", label="Выбросы")
    plt.scatter(range(len(data[~emissions])), data[~emissions])
    plt.xlabel("Географические зоны Калифорнии")
    plt.ylabel("Медианный доход")
    plt.title("Выбросы по IQR")
    plt.legend()
    plt.show()


def main():
    # Загружаем данные сразу в виде DataFrame
    data = fetch_california_housing(as_frame=True).frame

    # Берём медианный доход
    med = data["MedInc"]

    # Находим выбросы по признаку
    # Выбросы обозначены как True, все остальное False
    emissions = detect_emission(med, 1.5)
    if emissions.sum() == 0:
        print("Выбросов не найдено")
        return

    # Вывод выбросов
    print(med[emissions])

    # Проверка типа данных
    if not med.dtype.kind in "if":
        print("Для построения графика данные должны быть числовыми")
        return
    
    # Построение графика
    create_graph(med, emissions)


if __name__ == "__main__":
    main()