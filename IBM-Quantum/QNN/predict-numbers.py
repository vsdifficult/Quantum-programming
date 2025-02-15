import pennylane as qml
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Константы
NUM_QUBITS = 10  # Количество кубитов (matches MNIST classes)
BATCH_SIZE = 32  # Размер пакета данных
LEARNING_RATE = 0.01  # Скорость обучения
EPOCHS = 5  # Количество эпох

# Инициализация квантового устройства
dev = qml.device("default.qubit", wires=NUM_QUBITS)

# Определение квантовой цепочки
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Закодируем входные данные в начальные состояния кубитов
    for i in range(NUM_QUBITS):
        qml.RY(inputs[i], wires=i)
    
    # Добавим параметризованные врата
    for i in range(NUM_QUBITS):
        qml.RZ(weights[i], wires=i)
    
    # Измерение
    # Return expectation values for all 10 qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

# Функция для преобразования данных
def data_encoding(x):
    # Нормализация данных для соответствия диапазону [-π, π]
    x = x.view(-1)  # Преобразование в одномерный массив
    x = 2 * np.pi * x / 255.0  # Масштабирование от 0 до 2π
    return x[:NUM_QUBITS]  # Берем первые NUM_QUBITS значений

# Загрузка данных MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Определение классификатора
class QuantumClassifier(torch.nn.Module):
    def __init__(self):
        super(QuantumClassifier, self).__init__()
        # Веса для квантовой цепочки
        self.weights = torch.nn.Parameter(torch.randn(NUM_QUBITS, requires_grad=True))

    def forward(self, inputs):
        outputs = []
        for x in inputs:
            # Закодируем данные и выполним квантовую цепочку
            encoded_data = data_encoding(x)
            # Keep everything in PyTorch tensors
            q_output = quantum_circuit(encoded_data, self.weights)
            outputs.append(torch.tensor(q_output, dtype=torch.float32))
        
        # Stack outputs while maintaining computation graph
        return torch.stack(outputs).reshape(-1, NUM_QUBITS)

# Определение функции потерь и оптимизатора
model = QuantumClassifier()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Обучение модели
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Получаем выходы модели
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        # Вычисляем функцию потерь
        loss = criterion(outputs.float(), labels)

        # Обратное распространение ошибки
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Статистика
        total_loss += loss.item()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Тестирование модели
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
