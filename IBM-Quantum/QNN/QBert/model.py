import numpy as np
from qiskit.quantum_info import Statevector
from sklearn.preprocessing import normalize
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

# Функция для создания квантовых эмбеддингов
def create_quantum_embedding(vector):
    # Нормализуем вектор
    normalized_vector = normalize(vector.reshape(1, -1))[0]
    # Создаем квантовое состояние
    quantum_state = Statevector(normalized_vector)
    return quantum_state

# Функция для преобразования квантового состояния в классический вектор
def quantum_to_classical(quantum_state):
    # Получаем амплитуды квантового состояния
    classical_vector = quantum_state.data
    return classical_vector

# Кастомный класс для интеграции квантовых эмбеддингов с BERT
class QuantumBERT(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", embedding_dim=768):
        super(QuantumBERT, self).__init__()
        # Загружаем предобученную модель BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        # Линейное преобразование для адаптации квантовых эмбеддингов
        self.embedding_transform = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, texts):
        all_embeddings = []
        for text in texts:
            # Токенизация текста с паддингом
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",  # Добавляем паддинг до максимальной длины
                truncation=True,
                max_length=128  # Максимальная длина последовательности
            )
            input_ids = tokens["input_ids"][0]  # Берём первую последовательность
            attention_mask = tokens["attention_mask"][0]

            quantum_embeddings = []
            for token_id in input_ids:
                if token_id == 0:  # Пропускаем токены [PAD], если они есть
                    quantum_embeddings.append(np.zeros(768))  # Добавляем нулевой вектор для паддинга
                else:
                    # Получаем эмбеддинг токена из BERT
                    token_embedding = self.bert.embeddings.word_embeddings(token_id).detach().numpy()
                    # Создаем квантовое состояние
                    quantum_state = create_quantum_embedding(token_embedding)
                    # Преобразуем обратно в классический вектор
                    classical_vector = quantum_to_classical(quantum_state)
                    quantum_embeddings.append(classical_vector)

            # Преобразуем список в тензор
            quantum_embeddings = torch.tensor(np.array(quantum_embeddings), dtype=torch.float32)

            # Адаптируем размерность к BERT
            transformed_embeddings = self.embedding_transform(quantum_embeddings.unsqueeze(0))

            all_embeddings.append(transformed_embeddings)

        # Объединяем все эмбеддинги, учитывая одинаковую длину последовательностей
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Передаем эмбеддинги в BERT
        outputs = self.bert(inputs_embeds=all_embeddings, attention_mask=attention_mask.unsqueeze(0))
        return outputs.last_hidden_state

# Пример использования
if __name__ == "__main__":
    # Создаем модель
    model = QuantumBERT()

    # Исходные тексты
    texts = [
        "Quantum computing is fascinating",
        "Natural language processing is exciting"
    ]

    # Выполняем forward pass
    outputs = model(texts)

    # Выводим результаты
    print("BERT Output Shape:", outputs.shape)