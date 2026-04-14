import lib_new as lib
import numpy as np

cir = lib.Circuit(rows=5, cols=5)
cir.draw()
valid_circuit = cir.solve()
# cir.draw()