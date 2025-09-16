import matplotlib.pyplot as plt
# load_boston() устарел и больше не используется
from sklearn.datasets import fetch_california_housing


def detect_emission(data, coef=1.5):
    """Функция для поиска выбросов по IQR"""
    # Определение границ
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - coef * IQR
    upper = Q3 + coef * IQR
    # Числа вне границ являются выбросами
    return (data < lower) | (data > upper)


def create_graph(data, emisions):
    """Функция для построения графика"""
    plt.figure(figsize=(16, 8))
    plt.scatter(range(len(data[emisions])), data[emisions], color="red", label="Выбросы")
    plt.scatter(range(len(data[~emisions])), data[~emisions])
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
    emisions = detect_emission(med, 1.5)
    if emisions.sum() == 0:
        print("Выбросов не найдено")
        return

    # Вывод выбросов
    print(med[emisions])

    # Проверка типа данных
    if not med.dtype.kind in "if":
        print("Для построения графика данные должны быть числовыми")
        return
    
    # Построение графика
    create_graph(med, emisions)


if __name__ == "__main__":
    main()