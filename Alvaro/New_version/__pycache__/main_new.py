import lib_new as lib

# ─── Circuito monofásico ───────────────────────────────────────────────────
print(">>> Generando circuito monofásico...")
cir = lib.Circuit(rows=5, cols=5, freq=50, seed=42)

cir.draw()

ok = cir.solve()
if ok:
    cir.report()
    cir.draw_phasors()                    # tensiones + corrientes
    cir.draw_phasors(show_currents=False) # solo tensiones
else:
    print("No se pudo resolver el circuito.")

# ─── Circuito trifásico ────────────────────────────────────────────────────
print("\n>>> Generando circuito trifásico en Y...")
tp_y = lib.ThreePhaseCircuit(freq=50, v_line=400, connection='Y', seed=7)
tp_y.solve()
tp_y.report()
tp_y.draw_phasors()

print("\n>>> Generando circuito trifásico en Δ...")
tp_d = lib.ThreePhaseCircuit(freq=50, v_line=400, connection='delta', seed=7)
tp_d.solve()
tp_d.report()
