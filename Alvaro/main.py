import lib
import numpy as np

cir = lib.circuit(rows=2, cols=3)

print('Nodes:')
for n in cir.nodes:
    print(f'\t{n}')
print('Edges:')
for e in cir.edges:
    print(f'\t{e}')



