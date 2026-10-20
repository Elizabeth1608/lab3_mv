# Лабораторная работа 3. Численное интегрирование
# Вариант 9: f(x) = cos(x^2 * cos(x)), a = 5, b = 7

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Функция и пределы
def f(x):
    return np.cos(x**2 * np.cos(x))

a, b = 5.0, 7.0

# 2. Нахождение "точного" значения (эталона)
# Используем мощный алгоритм scipy для проверки
I_exact, _ = quad(f, a, b)

print("Лабораторная работа 3. Численное интегрирование")
print(f"Функция: f(x) = cos(x^2 * cos(x))")
print(f"Отрезок: [{a}, {b}]")
print(f"Эталонное значение (F(b) - F(a)): I_точн = {I_exact:.8f}\n")

# 3. Оценка максимумов производных (для теоретической погрешности)
def f_2nd_deriv(x, h=1e-4):
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

def f_4th_deriv(x, h=1e-3):
    return (f(x + 2*h) - 4*f(x + h) + 6*f(x) - 4*f(x - h) + f(x - 2*h)) / (h**4)

x_check = np.linspace(a, b, 2000)
M2 = np.max(np.abs(f_2nd_deriv(x_check)))
M4 = np.max(np.abs(f_4th_deriv(x_check)))

# 4. Параметры сетки (масштабируемость)
N = 600  # Кратно 2 (Симпсон) и 3 (3/8)
h = (b - a) / N
x = np.linspace(a, b, N + 1)
y = f(x)

print(f"Выбрано интервалов N = {N} (кратно 6)")
print(f"Величина шага на сетке: h = {h:.6f}\n")

# 5. Вычисления
# Прямоугольники
I_left = h * np.sum(y[:-1])
I_right = h * np.sum(y[1:])
x_mid = x[:-1] + h/2
I_mid = h * np.sum(f(x_mid))

# Трапеции
I_trap = h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)

# Симпсон (на сдвоенном интервале)
I_simp = (h/3) * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])

# Трех восьмых
I_38 = (3*h/8) * (y[0] + 3*np.sum(y[1:-1:3]) + 3*np.sum(y[2:-1:3]) + 2*np.sum(y[3:-1:3]) + y[-1])

# Гаусс (n=5) с разбиением на части для точности
K = 20 # количество подотрезков
h_gauss = (b - a) / K
I_gauss = 0
t_g = np.array([-0.90617985, -0.53846931, 0.0, 0.53846931, 0.90617985])
A_g = np.array([0.23692688, 0.47862868, 0.56888889, 0.47862868, 0.23692688])

for i in range(K):
    a_k = a + i*h_gauss
    b_k = a + (i+1)*h_gauss
    x_g = (b_k + a_k)/2 + (b_k - a_k)/2 * t_g
    I_gauss += (b_k - a_k)/2 * np.sum(A_g * f(x_g))

# 6. Оценка погрешностей (Совокупное/Пиковое отклонение)
# R(h) - теоретическая оценка
R_trap = (b - a) / 12 * M2 * h**2
R_simp = (b - a) / 180 * M4 * h**4
R_38 = (b - a) / 80 * M4 * h**4

# 7. Вывод в виде таблицы
print(f"{'Метод':<25} | {'Порядок O(h^m)':<15} | {'I_прибл':<15} | {'Погрешность |I_точн - I|':<20}")

# Функция для вывода строки таблицы
def print_row(name, order, I_approx):
    err = abs(I_exact - I_approx)
    print(f"{name:<25} | {order:<15} | {I_approx:<15.8f} | {err:<20.2e}")

print_row("Прямоугольники (левые)", "O(h)", I_left)
print_row("Прямоугольники (правые)", "O(h)", I_right)
print_row("Прямоугольники (средние)", "O(h^2)", I_mid)
print_row("Трапеции", "O(h^2)", I_trap)
print_row("Симпсона", "O(h^4)", I_simp)
print_row("Трех восьмых", "O(h^4)", I_38)
print_row("Гаусса (n=5)", "O(h^10)", I_gauss)

print("\nТеоретическая оценка остаточного члена (max |R(x)|):")
print(f"Для трапеций (R): {R_trap:.2e}")
print(f"Для Симпсона (R): {R_simp:.2e}")
print(f"Для 3/8 (R):      {R_38:.2e}")

# 8. Построение графика
plt.figure(figsize=(10, 5))
x_plot = np.linspace(a, b, 1000)
plt.plot(x_plot, f(x_plot), 'b-', label='f(x) = cos(x²cos(x))')
plt.fill_between(x_plot, f(x_plot), alpha=0.2, color='blue')
plt.title('Подынтегральная функция f(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()