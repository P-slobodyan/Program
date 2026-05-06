import torch as pt
import torch.nn as nn 
import torch.optim as optim

epochs = 500000       # Количество эпох
h = 0.5               # Шаг обучения (скорость)
error_epochs = 0.0001 # Допустимая ошибка

x = pt.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=pt.float32)   # Входные данные
y_ist = pt.tensor([[0], [1], [1], [1]], dtype=pt.float32)           # Истинный итог (для обучения)

class Neuron(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,1)
        #Fully Connected - полносвязный слой

    def forward(self,x):
        x = pt.sigmoid(self.fc1(x))
        return x

nerve = Neuron()

criterion = nn.MSELoss()                # Создание объекта класса MSELoss, для вычисления СКО
optimizer = optim.SGD(nerve.parameters(),lr=h)

incons_past = 1     # Прошлая ошибка
incons_d = 1        # Разница ошибок
incons_index = 0    # Число одинаковой разницы ошибок
incons_step = 2000  # Предел переменой incons_index

for index in range(epochs):
    sigm_fun = nerve(x)                 # Вычисление активационной функции
    incons = criterion(sigm_fun, y_ist) # Вычисление СКО (среднеквадратичная ошибка)

    # прервать, чтобы не вызвать переобучение
    if incons_past - incons.item() == incons_d:
        incons_index += 1
        if incons_index == incons_step:
            print(f"Число эпох: {index}")
            break
    elif incons.item() <= error_epochs:
        print(f"Число эпох: {index}")
        break
    else:
        incons_d = incons_past - incons.item()


    incons_past = incons.item()

    # Обновление весов
    optimizer.zero_grad()
    incons.backward()           # Вычисляем градиенты
    optimizer.step()

print(f"Выходы после обучения: {sigm_fun.data}")
print(f"Невязка после обучения: {incons.item()}")

del x, y_ist, nerve, criterion, optimizer, sigm_fun, incons