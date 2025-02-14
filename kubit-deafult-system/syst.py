from qiskit import QuantumCircuit, Aer, execute 
from qiskit.visualization import plot_histogram 

qc = QuantumCircuit(2, 2) 
qc.h(0) 
qc.cx(0, 1) 
qc.measure([0, 1], [0, 1]) 

simulator = Aer.get_backend("qasm_simulator") 
result = execute(qc, backend=simulator, shots=1000).result() 

counts = result.get_counts(qc) 
print(counts ) 
plot_histogram(counts)