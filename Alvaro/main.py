import lib
import numpy as np

circuit = lib.circuit(6)
circuit.draw()


print(circuit.edges)

# sol = circuit.solve()