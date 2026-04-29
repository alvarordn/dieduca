from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
import random
import numpy as np
import traceback
import re

from .alvaro.lib_new import Circuit, ThreePhaseCircuit


# ======================================================
# 🔧 HELPERS
# ======================================================

def complex_to_dict(z):
    if z is None:
        return {"re": 0.0, "im": 0.0}
    return {
        "re": float(np.real(z)),
        "im": float(np.imag(z))
    }


def safe_value(v):
    if isinstance(v, complex):
        return complex_to_dict(v)
    if isinstance(v, np.generic):
        return float(v)
    return v


def safe_float(v):
    if v is None:
        return 0.0
    if isinstance(v, complex):
        return float(abs(v))
    if isinstance(v, (int, float, np.number)):
        return float(v)
    return 0.0


def normalize_type(t):
    if not t:
        return "unknown"

    t = str(t).lower()

    if "res" in t:
        return "resistor"
    if "cap" in t:
        return "capacitor"
    if "ind" in t:
        return "inductor"
    if "volt" in t:
        return "source"
    if "wire" in t:
        return "wire"

    return t


# ======================================================
# 🧠 NODOS TYPE (FRONTEND IMAGES)
# ======================================================

def determinar_tipo_nodo(row, col, rows, cols):

    is_top = row == 0
    is_bottom = row == rows - 1
    is_left = col == 0
    is_right = col == cols - 1

    if is_top and is_left:
        return "corner-top-left"
    if is_top and is_right:
        return "corner-top-right"
    if is_bottom and is_left:
        return "corner-bottom-left"
    if is_bottom and is_right:
        return "corner-bottom-right"

    if is_top:
        return "edge-top"
    if is_bottom:
        return "edge-bottom"
    if is_left:
        return "edge-left"
    if is_right:
        return "edge-right"

    return "center"


# ======================================================
# 🚀 API
# ======================================================

@api_view(['POST'])
@permission_classes([AllowAny])
def generar_circuito(request):


    try:

        bloque_id = int(request.data.get('bloque', 1))
        rows = int(request.data.get('rows', 2))
        cols = int(request.data.get('cols', 3))

                # ======================================================
        # 🔺 TRIFÁSICO (MEJORADO)
        # ======================================================
        if bloque_id in [9, 10]:

            num_sections = int(request.data.get('num_sections', 3))

            circuit = ThreePhaseCircuit(
                num_sections=num_sections,
                freq=50,
                v_line=400,
                seed=random.randint(1, 9999)
            )

            circuit.solve()

            sections = []

            visual_options = ["series", "paraleloY", "paraleloDelta"]
            prev_visual = None

            # 🔥 SOLO UN LOOP (ARREGLADO)
            for i, s in enumerate(circuit.sections):

                ref = s["elements"]["A"]

                visual = random.choice(visual_options)

                # evita repetición inmediata
                if visual == prev_visual:
                    visual = random.choice(visual_options)

                prev_visual = visual

                kind = "serie" if visual == "series" else "paralelo"

                sections.append({
                    "idx": i,
                    "type": kind,
                    "visual": visual,
                    "label": ref.get("string", ""),

                    "Z_phase": complex_to_dict(s.get("Z_phase")),

                    "elements": {
                        ph: {
                            "type": normalize_type(
                                s["elements"][ph].get("element")
                            ),
                            "value": safe_value(
                                s["elements"][ph].get("value")
                            ),
                            "string": s["elements"][ph].get("string", "")
                        }
                        for ph in ["A", "B", "C"]
                    }
                })

            # =========================
            # RESULTS
            # =========================
            results = {}

            P_total = 0
            Q_total = 0
            S_total = 0

            for ph in ["A", "B", "C"]:
                r = circuit.results[ph]

                results[ph] = {
                    "V_phase": complex_to_dict(r["V_phase"]),
                    "I_line": complex_to_dict(r["I_line"]),
                    "P": float(r["P"]),
                    "Q": float(r["Q"]),
                    "S": float(r["S"])
                }

                P_total += abs(r["P"])
                Q_total += abs(r["Q"])
                S_total += abs(r["S"])

            params = {
                "freq": float(circuit.freq),
                "v_line": float(circuit.v_line),
                "P_total": float(P_total),
                "Q_total": float(Q_total),
                "S_total": float(S_total),
            }

            return Response({
                "success": True,
                "tipo": "trifasico",
                "circuito": {
                    "params": params,
                    "sections": sections,
                    "results": results
                }
            })

        # ======================================================
        # ⚡ MONOFÁSICO
        # ======================================================

        circuit = Circuit(rows=rows, cols=cols)
        circuit.solve()

        # -----------------------------
        # NODOS
        # -----------------------------
        nodos = []

        for node in circuit.G.nodes():

            match = re.match(r"N(\d)(\d)", node)
            if not match:
                continue

            row = int(match.group(1))
            col = int(match.group(2))

            nodos.append({
                "id": node,
                "row": row,
                "col": col,
                "type": determinar_tipo_nodo(row, col, rows, cols),
                "potential": safe_float(
                    circuit.G.nodes[node].get("potential", 0)
                )
            })

        # -----------------------------
        # COMPONENTES
        # -----------------------------
        componentes = []

        for i, (u, v) in enumerate(circuit.G.edges()):
            data = circuit.G[u][v]

            u_match = re.match(r"N(\d)(\d)", u)
            v_match = re.match(r"N(\d)(\d)", v)

            if u_match and v_match:
                ur, uc = int(u_match.group(1)), int(u_match.group(2))
                vr, vc = int(v_match.group(1)), int(v_match.group(2))

                orientation = "horizontal" if ur == vr else "vertical"
            else:
                orientation = "horizontal"

            componentes.append({
                "id": f"c{i}",
                "source": u,
                "target": v,

                "type": normalize_type(data.get("element")),

                "value": str(data.get("string", "")),

                # 🔥 NUEVO
                "orientation": orientation,

                "current": safe_float(data.get("current", 0)),
                "v_drop": safe_float(data.get("v_drop", 0))
            })

        return Response({
            "success": True,
            "tipo": "monofasico",
            "circuito": {
                "rows": rows,
                "cols": cols,
                "nodos": nodos,
                "componentes": componentes
            }
        })


    except Exception as e:
        print(traceback.format_exc())
        return Response({
            "success": False,
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)