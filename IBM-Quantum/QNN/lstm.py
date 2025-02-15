import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from keras import layers, models
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# Parameters
n_qubits = 4  # Number of qubits
vocab_size = 10000  # Vocabulary size
embedding_dim = 64  # Embedding dimension
max_length = 100  # Maximum sequence length

# Quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum layer
@qml.qnode(dev)
def quantum_layer(inputs):
    # Ensure inputs have correct shape
    if len(inputs) != n_qubits:
        raise ValueError(f"Input must have exactly {n_qubits} elements")
    
    # Encode inputs into quantum state
    for i in range(n_qubits):
        qml.RY(float(inputs[i]), wires=i)
    
    # Simple quantum circuit
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    # Measure expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Classical neural network
def create_classical_model():
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.LSTM(n_qubits, return_sequences=False),  # Output n_qubits values directly
    ])
    return model

# Hybrid model
def process_batch(classical_output):
    # Process batch through quantum layer
    quantum_outputs = []
    for i in range(classical_output.shape[0]):
        # Extract and process each sample
        sample = classical_output[i].numpy()
        quantum_output = quantum_layer(sample)
        quantum_outputs.append(quantum_output)
    return tf.convert_to_tensor(quantum_outputs, dtype=tf.float32)

def hybrid_model(inputs):
    # Apply classical neural network
    classical_output = classical_model(inputs)
    
    # Process batch through quantum layer
    return process_batch(classical_output)

# Create model
classical_model = create_classical_model()

# Initialize model weights
classical_model.build(input_shape=(None, max_length))

# Create hybrid model without tf.function decorator
# to avoid graph execution issues

# Example data
texts = ["This is a good movie", "I did not like this film", "The plot was amazing"]
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, dtype=np.str_)

# Convert and reshape inputs to match model expectations
inputs = np.array(padded_sequences, dtype=np.float32)
inputs = tf.reshape(inputs, [inputs.shape[0], max_length])  # Ensure proper input shape

# Initialize model weights
classical_model.build(input_shape=(None, max_length))

# Apply hybrid model
output = hybrid_model(inputs)
print(output)
