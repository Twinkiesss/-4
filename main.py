import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Лабораторная работа № 4. Метод k-ближайших соседей.
# Используются NumPy и Pandas для обработки данных и реализации k-NN.
# Matplotlib используется только для визуализации.

CSV_FILE = "wine_dataset.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.25
K_VALUES = [1, 3, 5, 7, 10, 15]


def load_data(path):
    df = pd.read_csv(path)
    # Поддержка стандартного UCI wine.data без заголовков:
    if df.shape[1] == 14 and "Class" not in df.columns:
        names = ["Class", "Alcohol", "Malic Acid", "Ash", "Alcalinity of ash",
                 "Magnesium", "Total phenols", "Flavanoids",
                 "Nonflavanoid phenols", "Proanthocyanins", "Color intensity",
                 "Hue", "OD280/OD315 of diluted wines", "Proline"]
        df.columns = names
    # Удаляем полностью пустые строки/столбцы, которые иногда встречаются
    # в неправильно экспортированных CSV.
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if "Class" not in df.columns:
        # Вариант с колонкой класса под другим распространённым названием
        possible = [c for c in df.columns if str(c).strip().lower() in
                    ("class", "target", "cultivar", "wine", "type")]
        if possible:
            df = df.rename(columns={possible[0]: "Class"})
    if "Class" not in df.columns:
        raise ValueError("Не найдена целевая колонка Class. Нужен CSV с меткой класса вина.")
    df = df.dropna(subset=["Class"]).copy()
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df = df.dropna(subset=["Class"])
    df["Class"] = df["Class"].astype(int)
    for col in df.columns:
        if col != "Class":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Удаляем столбцы, полностью состоящие из пропусков.
    df = df.dropna(axis=1, how="all")
    if len(df) == 0:
        raise ValueError("В CSV нет корректных наблюдений. Проверьте файл данных.")
    return df


def train_test_split_manual(X, y, test_size=0.25, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    # Разделение отдельно внутри каждого класса сохраняет пропорции классов.
    for label in np.unique(y):
        ids = np.where(y == label)[0]
        rng.shuffle(ids)
        n_test = max(1, int(round(len(ids) * test_size)))
        test_idx.extend(ids[:n_test])
        train_idx.extend(ids[n_test:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def impute_and_scale(X_train, X_test):
    # Медианы вычисляются только по обучающей выборке.
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    train = np.where(np.isnan(X_train), medians, X_train)
    test = np.where(np.isnan(X_test), medians, X_test)
    # Стандартизация: параметры берутся только из train.
    means = train.mean(axis=0)
    stds = train.std(axis=0)
    stds[stds == 0] = 1.0
    return (train - means) / stds, (test - means) / stds


def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for point in X_test:
        distances = np.sqrt(np.sum((X_train - point) ** 2, axis=1))
        nearest = np.argsort(distances)[:k]
        labels, counts = np.unique(y_train[nearest], return_counts=True)
        # При равенстве голосов np.unique возвращает метки по возрастанию;
        # argmax детерминированно выбирает первую из них.
        predictions.append(labels[np.argmax(counts)])
    return np.asarray(predictions)


def confusion_matrix_manual(y_true, y_pred, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    mapping = {label: i for i, label in enumerate(labels)}
    for actual, predicted in zip(y_true, y_pred):
        matrix[mapping[actual], mapping[predicted]] += 1
    return matrix


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def plot_confusion(matrix, labels, title, filename):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.show()


def main():
    df = load_data(CSV_FILE)
    print("Размер исходных данных:", df.shape)
    print("\nПервые строки:")
    print(df.head())
    print("\nПропуски до обработки:")
    print(df.isna().sum())
    print("\nСтатистика признаков:")
    print(df.drop(columns=["Class"]).describe().T.round(3))
    print("\nРаспределение классов:")
    print(df["Class"].value_counts().sort_index())

    # Для визуализаций и моделей оставляем числовые признаки.
    feature_cols = [c for c in df.columns if c != "Class"]
    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df["Class"].to_numpy(dtype=int)

    # Графическая статистика: boxplot и гистограммы.
    fig, ax = plt.subplots(figsize=(13, 6))
    df[feature_cols].boxplot(ax=ax, rot=70)
    ax.set_title("Распределение значений признаков вина")
    ax.set_ylabel("Исходные значения")
    fig.tight_layout()
    fig.savefig("01_boxplot.png", dpi=160)
    plt.show()

    df[feature_cols].hist(figsize=(14, 10), bins=15, edgecolor="black")
    plt.suptitle("Гистограммы химических признаков")
    plt.tight_layout()
    plt.savefig("02_histograms.png", dpi=160)
    plt.show()

    # 3D-визуализация трёх химических признаков.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    x_name, y_name, z_name = "Alcohol", "Flavanoids", "Color intensity"
    for label in np.unique(y_all):
        mask = y_all == label
        ax.scatter(df.loc[mask, x_name], df.loc[mask, y_name],
                   df.loc[mask, z_name], label=f"Класс {label}", s=28)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)
    ax.set_title("3D-визуализация образцов вина")
    ax.legend()
    plt.tight_layout()
    plt.savefig("03_3d.png", dpi=160)
    plt.show()

    # Модель 1: случайный набор признаков (фиксируется seed для воспроизводимости).
    rng = np.random.default_rng(RANDOM_STATE)
    count_random = min(5, len(feature_cols))
    random_features = list(rng.choice(feature_cols, size=count_random, replace=False))
    # Модель 2: заранее заданный набор признаков с химической интерпретацией.
    fixed_features = ["Alcohol", "Flavanoids", "Color intensity", "Proline"]
    fixed_features = [f for f in fixed_features if f in feature_cols]
    print("\nПризнаки модели 1 (случайный выбор):", random_features)
    print("Признаки модели 2 (фиксированный набор):", fixed_features)

    labels = np.unique(y_all)
    all_results = []

    for model_name, selected in [("Модель 1 — случайные признаки", random_features),
                                 ("Модель 2 — фиксированные признаки", fixed_features)]:
        X = df[selected].to_numpy(dtype=float)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split_manual(
            X, y_all, TEST_SIZE, RANDOM_STATE)
        X_train, X_test = impute_and_scale(X_train_raw, X_test_raw)
        max_k = len(y_train)
        for k in K_VALUES:
            if k > max_k:
                continue
            y_pred = knn_predict(X_train, y_train, X_test, k)
            score = accuracy(y_test, y_pred)
            matrix = confusion_matrix_manual(y_test, y_pred, labels)
            all_results.append({"Модель": model_name, "k": k, "Accuracy": score})
            print(f"\n{model_name}; k={k}; accuracy={score:.4f}")
            print("Матрица ошибок (строки — истинный класс, столбцы — предсказанный):")
            print(pd.DataFrame(matrix, index=labels, columns=labels))
            safe_name = "model1" if model_name.startswith("Модель 1") else "model2"
            plot_confusion(matrix, labels,
                           f"{model_name}, k={k}",
                           f"confusion_{safe_name}_k{k}.png")

    results = pd.DataFrame(all_results)
    print("\nСравнение качества:")
    print(results.to_string(index=False))
    results.to_csv("knn_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in results["Модель"].unique():
        part = results[results["Модель"] == name]
        ax.plot(part["k"], part["Accuracy"], marker="o", label=name)
    ax.set_xlabel("Число соседей k")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Зависимость качества классификации от k")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("04_accuracy_by_k.png", dpi=160)
    plt.show()

    best = results.loc[results.groupby("Модель")["Accuracy"].idxmax()]
    print("\nЛучшее k для каждой модели:")
    print(best.to_string(index=False))
    print("\nГотово. Графики и таблица результатов сохранены в текущей папке.")


if __name__ == "__main__":
    main()
