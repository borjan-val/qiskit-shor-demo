# See https://learning.quantum.ibm.com/course/fundamentals-of-quantum-algorithms/phase-estimation-and-factoring#implementation-in-qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library import QFT
import numpy as np
from math import gcd, floor, log
from fractions import Fraction
import random
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
import os

print("Connecting to IBM Quantum Qiskit Runtime Service...")
service = QiskitRuntimeService(channel="ibm_quantum", token=os.getenv("IBM_QUANTUM_TOKEN"))
print("Successful.")

qpu_time = 0.0
qpu_jobs = 0

def mod_mult_gate(b,N):
    if gcd(b,N)>1:
        print(f"Error: gcd({b},{N}) > 1")
    else:
        n = floor(log(N-1,2)) + 1
        U = np.full((2**n,2**n),0)
        for x in range(N): U[b*x % N][x] = 1
        for x in range(N,2**n): U[x][x] = 1
        G = UnitaryGate(U)
        G.name = f"M_{b}"
        return G

def order_finding_circuit(a,N):
    if gcd(a,N)>1:
        print(f"Error: gcd({a},{N}) > 1")
    else:
        n = floor(log(N-1,2)) + 1
        m = 2*n

        control = QuantumRegister(m, name = "X")
        target = QuantumRegister(n, name = "Y")
        output = ClassicalRegister(m, name = "Z")
        circuit = QuantumCircuit(control, target, output)

        # Initialize the target register to the state |1>
        circuit.x(m)

        # Add the Hadamard gates and controlled versions of the
        # multiplication gates
        for k, qubit in enumerate(control):
            circuit.h(k)
            b = pow(a,2**k,N)
            circuit.compose(
                mod_mult_gate(b,N).control(),
                qubits = [qubit] + list(target),
                inplace=True)

        # Apply the inverse QFT to the control register
        circuit.compose(
            QFT(m, inverse=True),
            qubits=control,
            inplace=True)

        # Measure the control register
        circuit.measure(control, output)

        return circuit

def find_order(a,N):
    global qpu_jobs, qpu_time

    if gcd(a,N)>1:
        print(f"Error: gcd({a},{N}) > 1")
    else:
        n = floor(log(N-1,2)) + 1
        m = 2*n
        print("Building quantum circuit...")
        circuit = order_finding_circuit(a,N)
        print("Successful.")

        # Retrieve QPU backend
        print("Retrieving QPU backend from IBM Quantum...")
        backend = service.least_busy(simulator=False, operational=True)
        print("Found backend \"" + backend.name + "\".")

        print("Starting circuit transpilation...")
        transpiled_circuit = transpile(circuit,backend)
        print("Successful.")

        sampler = Sampler(mode=backend)

        while True:
            job = sampler.run(
                [transpiled_circuit],
                shots=1)
            
            qpu_jobs += 1
            qpu_time += job.usage()

            result = job.result()[0].data.Z.get_bitstrings()
            y = int(result[0],2)
            r = Fraction(y/2**m).limit_denominator(N).denominator
            if pow(a,r,N)==1: break
        return r

N = int(input("Integer to factor: "))

FACTOR_FOUND = False

# First we'll check to see if N is even or a nontrivial power.
# Order finding won't help for factoring a *prime* power, but
# we can easily find a nontrivial factor of *any* nontrivial
# power, whether prime or not.

if N % 2 == 0:
    print("Even number")
    d = 2
    FACTOR_FOUND = True
else:
    for k in range(2,round(log(N,2))+1):
        d = int(round(N ** (1/k)))
        if d**k == N:
            FACTOR_FOUND = True
            print("Number is a power")
            break

# Now we'll iterate until a nontrivial factor of N is found.

while not FACTOR_FOUND:
    a = random.randint(2,N-1)
    d = gcd(a,N)
    if d>1:
        FACTOR_FOUND = True
        print(f"Lucky guess of {a} modulo {N}")
    else:
        r = find_order(a,N)
        print(f"The order of {a} modulo {N} is {r}")
        if r % 2 == 0:
            x = pow(a,r//2,N) - 1
            d = gcd(x,N)
            if d>1: FACTOR_FOUND = True

print(f"Factor found: {d}")
print(f"Took {qpu_time} seconds QPU time distributed over {qpu_jobs} QPU jobs.")