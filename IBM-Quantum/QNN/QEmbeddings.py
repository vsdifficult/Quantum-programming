import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.feature_extraction.text import TfidfVectorizer

# Функция для нормализации вектора
def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

# Функция для расширения вектора до ближайшей степени двойки
def pad_to_power_of_two(vector):
    current_length = len(vector)
    next_power_of_two = 1
    while next_power_of_two < current_length:
        next_power_of_two *= 2
    if next_power_of_two > current_length:
        padded_vector = np.zeros(next_power_of_two)
        padded_vector[:current_length] = vector
    else:
        padded_vector = vector
    return padded_vector

# Функция для преобразования вектора в квантовое состояние
def vector_to_quantum_state(vector):
    # Расширяем вектор до ближайшей степени двойки
    padded_vector = pad_to_power_of_two(vector)
    # Нормализуем вектор
    normalized_vector = normalize_vector(padded_vector)
    # Создаем квантовое состояние
    quantum_state = Statevector(normalized_vector)
    return quantum_state

# Функция для создания квантовой цепочки из состояния
def create_quantum_circuit_from_state(quantum_state):
    # Определяем количество кубитов
    num_qubits = int(np.ceil(np.log2(len(quantum_state))))
    # Создаем квантовую цепочку
    qc = QuantumCircuit(num_qubits)
    # Инициализируем цепочку заданным состоянием
    qc.initialize(quantum_state, range(num_qubits))
    return qc

# Функция для конвертации текста в квантовые эмбеддинги
def text_to_quantum_embeddings(texts):
    # Используем TF-IDF для получения векторного представления текста
    vectorizer = TfidfVectorizer()
    vectorized_texts = vectorizer.fit_transform(texts).toarray()
    
    quantum_circuits = []
    for vector in vectorized_texts:
        # Преобразуем вектор в квантовое состояние
        quantum_state = vector_to_quantum_state(vector)
        # Создаем квантовую цепочку
        qc = create_quantum_circuit_from_state(quantum_state)
        quantum_circuits.append(qc)
    
    return quantum_circuits

# Пример использования
if __name__ == "__main__":
    # Набор текстов для обработки
    texts = [
        "Quantum computing is fascinating",
        "Natural language processing is exciting",
        "Machine learning and AI are powerful"
    ]
    
    # Конвертируем тексты в квантовые эмбеддинги
    quantum_circuits = text_to_quantum_embeddings(texts)
    
    # Выводим квантовые цепочки
    for i, qc in enumerate(quantum_circuits):
        print(f"Quantum Circuit for Text {i + 1}:")
        print(qc.draw())