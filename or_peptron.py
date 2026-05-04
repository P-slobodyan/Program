import torch as pt
import torch.nn as nn 

epochs = 500000       # Количество эпох
h = 0.5               # Шаг обучения (скорость)
error_epochs = 0.0001 # Допустимая ошибка

offset = pt.tensor([-0.101], requires_grad=True)                    # Смещение
w = pt.tensor([[0.75], [0.53]], requires_grad=True)                 # Веса
x = pt.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=pt.float32)   # Входные данные
y_ist = pt.tensor([[0], [1], [1], [1]], dtype=pt.float32)           # Истинный итог (для обучения)

for index in range(epochs):
    in_weig = x @ w + offset            # Перемножение водных данных и весов
    sigm_fun = pt.sigmoid(in_weig)      # Вычисление активационной функции
    criterion = nn.MSELoss()            # Создание объекта класса MSELoss для вычисления СКО
    incons = criterion(sigm_fun, y_ist) # Вычисление СКО (среднеквадратичная ошибка)

    # прервать, чтобы не вызвать переобучение
    if incons.item() <= error_epochs:
        print(f"Число эпох: {index}")
        break

    # Обновление весов
    incons.backward()           # Вычисляем градиенты
    with pt.no_grad():
        w -= h * w.grad 
        w.grad.zero_()          # Обнуляем градиенты после обновления весов
        offset -= h * offset.grad 
        offset.grad.zero_()     # Обнуляем градиенты после обновления смещения

print(f"Выходы после обучения: {sigm_fun.data}")
print(f"Невязка после обучения: {incons.item()}")