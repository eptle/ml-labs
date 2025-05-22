import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn import datasets
from sklearn.linear_model import LinearRegression


def get_csv_data():
	default_file = "lab1/src/student_scores.csv"
	path = input(f"Введите путь к CSV файлу (по умолчанию '{default_file}'): ").strip()
	if path == "":
		path = default_file
	try:
		return pd.read_csv(path)
	except Exception as err:
		print("Ошибка при чтении файла:", err)
		return None


def select_columns(df):
	cols = list(df.columns)
	print("Найденные столбцы:", cols)
	x_col = input("Введите название столбца для X: ").strip()
	y_col = input("Введите название столбца для Y: ").strip()
	if x_col not in df.columns or y_col not in df.columns:
		if len(cols) == 2:
			if x_col == "":
				x_col = cols[0]
			if y_col == "":
				y_col = cols[1]
		else:
			print("Один из указанных столбцов отсутствует в файле.")
			return None, None

	try:
		x = df[x_col].values
		y = df[y_col].values
	except Exception as err:
		print("Один из указанных столбцов отсутствует в файле.", err)
		return None, None
	return x_col, y_col


def print_stats(arr, name):
	print(f"\nСтатистика для столбца '{name}':\n"
		  f"  Количество: {len(arr)}\n"
		  f"  Минимум: {np.min(arr):.4f}\n"
		  f"  Максимум: {np.max(arr):.4f}\n"
		  f"  Среднее: {np.mean(arr):.4f}")


def plot_scatter(x, y, title, x_label, y_label):
	plt.figure(title)
	plt.scatter(x, y, color='blue', label='Данные')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend()


def compute_regression(x, y):
	n = len(x)
	mean_x = np.mean(x)
	mean_y = np.mean(y)
	num = 0
	den = 0
	for xi, yi in zip(x, y):
		num += (xi - mean_x) * (yi - mean_y)
		den += (xi - mean_x) ** 2
	if den == 0:
		print("Ошибка: деление на ноль при вычислении регрессии.")
		return None, None
	m = num / den
	b = mean_y - m * mean_x
	return m, b

def plot_regression_line(x, y, m, b, title, x_label, y_label):
	x_line = np.linspace(np.min(x), np.max(x), 100)
	y_line = m * x_line + b
	plt.figure(title)
	plt.scatter(x, y, color='blue', label='Данные')
	plt.plot(x_line, y_line, color='red', label='Регрессионная прямая')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend()
	return x_line, y_line


def plot_error_squares(x, y, m, b, x_line, y_line, title, x_label, y_label):
	plt.figure(title)
	ax = plt.gca()
	plt.scatter(x, y, color='blue', label='Данные')
	plt.plot(x_line, y_line, color='red', label='Регрессионная прямая')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	for xi, yi in zip(x, y):
		y_pred = m * xi + b
		error = yi - y_pred
		abs_error = abs(error)
		if abs_error == 0:
			continue
		square_x = xi
		square_y = min(yi, y_pred)

		rect = patches.Rectangle(
			(square_x - abs_error / 2, square_y),
			abs_error,
			abs_error,
			linewidth=1,
			edgecolor='green',
			facecolor='green',
			alpha=0.3
		)
		ax.add_patch(rect)
	plt.legend()


# чтоб квадраты были квадратами нужно раскомментировать строку ниже, но тогда график сожмется по x
# plt.gca().set_aspect('equal', adjustable='box')

# --- Новые функции для задачи с датасетом Diabetes ---

def plot_diabetes_regression(X_feature_values, y_target_values, feature_name, sklearn_model, m_custom, b_custom):
	plt.figure(figsize=(12, 7))
	plt.scatter(X_feature_values, y_target_values, color='blue', alpha=0.6, label='Фактические данные')

	x_line = np.linspace(np.min(X_feature_values), np.max(X_feature_values), 100).reshape(-1, 1)

	y_pred_sklearn_line = sklearn_model.predict(x_line)
	plt.plot(x_line, y_pred_sklearn_line, color='red', linewidth=2,
			 label=f'Регрессия Scikit-learn (m={sklearn_model.coef_[0]:.2f}, b={sklearn_model.intercept_:.2f})')

	y_pred_custom_line = m_custom * x_line.flatten() + b_custom
	plt.plot(x_line.flatten(), y_pred_custom_line, color='green', linestyle='--', linewidth=2,
			 label=f'Регрессия (свой алгоритм) (m={m_custom:.2f}, b={b_custom:.2f})')

	plt.xlabel(f"Признак: {feature_name}")
	plt.ylabel("Прогрессирование диабета (целевая переменная)")
	plt.title(f"Линейная регрессия: {feature_name} vs. Прогрессирование диабета")
	plt.legend()
	plt.grid(True)
	plt.show()


def diabetes_regression_task():
	diabetes = datasets.load_diabetes()
	X_all_features = diabetes.data
	y_target = diabetes.target
	feature_names = diabetes.feature_names

	print("--- Исследование набора данных Diabetes ---")
	print("\nОписание набора данных (первые 500 символов):")
	print(diabetes.DESCR[:500] + "...\n")

	print("Доступные признаки для выбора:")
	for i, name in enumerate(feature_names):
		print(f"  {i}: {name}")

	selected_feature_idx = -1
	default_feature_name = 'bmi'
	default_feature_idx = feature_names.index(default_feature_name)

	while selected_feature_idx < 0 or selected_feature_idx >= len(feature_names):
		try:
			choice = input(
				f"Введите индекс признака для X (0-{len(feature_names) - 1}), например, {default_feature_idx} для '{default_feature_name}' (оставьте пустым для '{default_feature_name}'): ").strip()
			if not choice:
				selected_feature_idx = default_feature_idx
				print(f"Ввод отсутствует, выбран признак по умолчанию: '{feature_names[selected_feature_idx]}'.")
				break
			selected_feature_idx = int(choice)
			if not (0 <= selected_feature_idx < len(feature_names)):
				print(f"Неверный индекс. Пожалуйста, введите число от 0 до {len(feature_names) - 1}.")
				selected_feature_idx = -1
		except ValueError:
			print("Неверный ввод. Пожалуйста, введите число.")
		except IndexError:
			print(f"Ошибка при выборе признака по умолчанию. Пожалуйста, введите индекс вручную.")

	selected_feature_name = feature_names[selected_feature_idx]

	X_selected_feature_values = X_all_features[:, selected_feature_idx]

	print(f"\nВыбранный признак для X: {selected_feature_name}")
	print_stats(X_selected_feature_values, selected_feature_name)
	print_stats(y_target, "Прогрессирование диабета (цель)")

	X_sklearn_input = X_selected_feature_values.reshape(-1, 1)

	print("\n--- Линейная регрессия с использованием Scikit-Learn ---")
	sklearn_model = LinearRegression()
	sklearn_model.fit(X_sklearn_input, y_target)
	sklearn_m = sklearn_model.coef_[0]
	sklearn_b = sklearn_model.intercept_
	y_pred_sklearn = sklearn_model.predict(X_sklearn_input)

	print("Значения коэффициентов (Scikit-Learn):")
	print(f"  Наклон (m): {sklearn_m:.4f}")
	print(f"  Смещение (b): {sklearn_b:.4f}")

	print("\n--- Линейная регрессия с использованием собственного алгоритма ---")
	custom_m, custom_b = compute_regression(X_selected_feature_values, y_target)
	if custom_m is None or custom_b is None:
		print("Не удалось рассчитать регрессию собственным алгоритмом.")
		return
	y_pred_custom = custom_m * X_selected_feature_values + custom_b

	print("Значения коэффициентов (собственный алгоритм):")
	print(f"  Наклон (m): {custom_m:.4f}")
	print(f"  Смещение (b): {custom_b:.4f}")

	print("\nОтрисовка данных и регрессионных прямых...")
	plot_diabetes_regression(X_selected_feature_values, y_target, selected_feature_name, sklearn_model, custom_m,
							 custom_b)

	print("\n--- Таблица с результатами предсказаний (первые 20 строк) ---")
	predictions_df = pd.DataFrame({
		'Фактическое значение': y_target,
		f'Предсказание Sklearn ({selected_feature_name})': y_pred_sklearn,
		f'Предсказание (свой алгоритм, {selected_feature_name})': y_pred_custom
	})
	pd.set_option('display.float_format', lambda x: '%.2f' % x)
	print(predictions_df.head(20).to_string())
	pd.reset_option('display.float_format')


def main_csv():
	df = get_csv_data()
	if df is None:
		return

	x_col, y_col = select_columns(df)
	if x_col is None or y_col is None:
		return

	x = df[x_col].values
	y = df[y_col].values

	print_stats(x, x_col)
	print_stats(y, y_col)

	plot_scatter(x, y, "Исходные точки", x_col, y_col)

	m, b = compute_regression(x, y)
	if m is None:
		return

	print(f"\nПараметры регрессионной прямой:\n"
		  f"Наклон (slope) = {m}\n"
		  f"Сдвиг (intercept) = {b}")

	x_line, y_line = plot_regression_line(x, y, m, b, "Регрессионная прямая", x_col, y_col)
	plot_error_squares(x, y, m, b, x_line, y_line, "Квадраты ошибок", x_col, y_col)
	plt.show()


if __name__ == "__main__":
	while True:
		print("\nКакую задачу вы хотите запустить?")
		print("1: Линейная регрессия на основе CSV-файла (студенческие баллы)")
		print("2: Линейная регрессия на наборе данных Diabetes (Scikit-Learn)")
		choice = input("Введите номер задачи (1 или 2): ").strip()

		if choice == '1':
			print("\n--- Запуск задачи: Линейная регрессия на основе CSV-файла ---")
			main_csv()
			break
		elif choice == '2':
			print("\n--- Запуск задачи: Линейная регрессия на наборе данных Diabetes ---")
			diabetes_regression_task()
			break
		else:
			print("Неверный выбор. Пожалуйста, введите 1 или 2.")
