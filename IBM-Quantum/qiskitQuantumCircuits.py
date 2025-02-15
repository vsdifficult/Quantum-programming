from qiskit import QuantumCircuit
from qiskit.primitives import Sampler  # Используем новый примитив Sampler
from qiskit.visualization import plot_histogram

qc = QuantumCircuit(3, 3)
qc.h(range(3))  # Создание суперпозиции для всех трёх кубитов
qc.measure([0, 1, 2], [0, 1, 2])  # Измерение всех трёх кубитов

# Вывод схемы цепочки
print(qc.draw())

# Шаг 4: Выполнение цепочки с использованием Sampler
sampler = Sampler()  # Создаём экземпляр Sampler
job = sampler.run(qc, shots=1000)  # Запускаем цепочку 1000 раз
result = job.result()

# Шаг 5: Получение и отображение результатов
counts = result.quasi_dists[0].binary_probabilities()  # Получаем распределение вероятностей
print("Результаты измерений:", counts)

# Опционально: Построение гистограммы результатов
plot_histogram(counts)