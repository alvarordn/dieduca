import lib_new as lib
import numpy as np

cir = lib.Circuit(rows=2, cols=3)
valid_circuit = cir.solve()
# cir.draw()