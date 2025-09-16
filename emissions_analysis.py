import matplotlib.pyplot as plt
# load_boston() устарел и больше не используется
from sklearn.datasets import fetch_california_housing


# Функция для поиска выбросов по IQR
def detect_emission(data):
    # Определение границ
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # Числа вне границ являются выбросами
    return (data < lower) | (data > upper)


# Функция для построения графика
def create_graph(data, emmisions):
    plt.figure(figsize=(10,6))
    plt.scatter(range(len(data[emmisions])), data[emmisions], color="red", label="Выбросы")
    plt.scatter(range(len(data[~emmisions])), data[~emmisions])
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
    emmisions = detect_emission(med)
    
    # Вывод выбросов
    print(med[emmisions])

    # Построение графика
    create_graph(med, emmisions)

if __name__ == "__main__":
    main()