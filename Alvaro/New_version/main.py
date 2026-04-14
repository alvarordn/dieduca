import lib

# ─── Circuito monofásico ───────────────────────────────────────────────────
# print(">>> Generando circuito monofásico...")
# cir = lib.Circuit(rows=2, cols=2, freq=50, seed=42)

# cir.draw()

# ok = cir.solve()
# if ok:
#     cir.report()
#     cir.draw_phasors() 
# else:
#     print("No se pudo resolver el circuito.")

# ─── Circuito trifásico ────────────────────────────────────────────────────
print("\n>>> Generando circuito trifásico (3 secciones, seed=7)...")
tp = lib.ThreePhaseCircuit(num_sections=5, freq=50, v_line=400, seed=71)
tp.draw()          # esquema unifilar
 
tp.solve()
tp.report()
