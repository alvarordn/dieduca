from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
import random
import numpy as np
import traceback
import re

# Importo las clases que me ha dado el profesor (NO se tocan)
from .alvaro.lib_new import Circuit, ThreePhaseCircuit


# =========================================================
# 🔁 CONVERSIÓN DE NÚMEROS COMPLEJOS A JSON
# =========================================================
# Esto lo hago porque el frontend (Angular) no entiende complejos tipo Python
# Entonces los separo en parte real e imaginaria
def complex_to_dict(z):
    if z is None:
        return {"re": 0.0, "im": 0.0}

    return {
        "re": float(np.real(z)),
        "im": float(np.imag(z))
    }


# =========================================================
# 🧼 NORMALIZAR VALORES PARA QUE NO ROMPAN EL FRONTEND
# =========================================================
# Aquí me aseguro de que TODO lo que mando sea JSON válido
def safe_value(v):

    if v is None:
        return 0.0

    # Si es complejo numpy lo convierto
    if isinstance(v, complex):
        return complex_to_dict(v)

    # Si es tipo numpy raro lo paso a float
    if isinstance(v, np.generic):
        return float(v)

    # Si es string lo limpio
    if isinstance(v, str):
        return v.strip()

    return v


# =========================================================
# 🔢 CONVERSIÓN SEGURA A FLOAT
# =========================================================
# Esto evita errores típicos de backend con valores raros o None
def safe_float(v):

    if v is None:
        return 0.0

    # Si es complejo, me quedo con el módulo
    if isinstance(v, complex):
        v = abs(v)

    # numpy number -> float normal
    if isinstance(v, np.number):
        v = float(v)

    try:
        v = float(v)
    except:
        return 0.0

    # elimino valores absurdamente pequeños (ruido numérico)
    if abs(v) < 1e-12:
        return 0.0

    # redondeo para no devolver 0.00000000003 cosas raras
    return round(v, 6)


# =========================================================
# 🧠 NORMALIZAR TIPOS DE COMPONENTES
# =========================================================
# Aquí traduzco lo que viene del lib.py a algo estándar
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

    # fuentes de corriente y tensión
    if "c_source" in t or "current" in t:
        return "c_source"
    if "v_source" in t or "volt" in t:
        return "v_source"

    if "wire" in t:
        return "wire"

    return t


# =========================================================
# 🧹 LIMPIEZA EXTRA DE FLOATS
# =========================================================
# Similar a safe_float pero más simple (fallback)
def clean_float(v, eps=1e-6):

    if v is None:
        return 0.0

    try:
        v = float(v)

        # si es casi 0, lo pongo directamente a 0
        if abs(v) < eps:
            return 0.0

        return v

    except:
        return 0.0


# =========================================================
# 📍 TIPO DE NODO SEGÚN SU POSICIÓN EN LA MATRIZ
# =========================================================
# Esto es solo para pintar el circuito en frontend bonito
def determinar_tipo_nodo(row, col, rows, cols):

    is_top = row == 0
    is_bottom = row == rows - 1
    is_left = col == 0
    is_right = col == cols - 1

    # esquinas
    if is_top and is_left:
        return "corner-top-left"
    if is_top and is_right:
        return "corner-top-right"
    if is_bottom and is_left:
        return "corner-bottom-left"
    if is_bottom and is_right:
        return "corner-bottom-right"

    # bordes
    if is_top:
        return "edge-top"
    if is_bottom:
        return "edge-bottom"
    if is_left:
        return "edge-left"
    if is_right:
        return "edge-right"

    # centro
    return "center"


# =========================================================
# 🚀 API PRINCIPAL: GENERAR CIRCUITO
# =========================================================
@api_view(['POST'])
@permission_classes([AllowAny])
def generar_circuito(request):

    try:

        # =========================
        # 📥 DATOS DEL FRONTEND
        # =========================
        bloque_id = int(request.data.get('bloque', 1))
        rows = int(request.data.get('rows', 2))
        cols = int(request.data.get('cols', 3))


        # =====================================================
        # ⚡ CASO TRIFÁSICO (BLOQUES 9 Y 10)
        # =====================================================
        if bloque_id in [9, 10]:

            num_sections = int(request.data.get('num_sections', 3))

            # Creo circuito trifásico usando la librería del profe
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

            # genero secciones del circuito
            for i, s in enumerate(circuit.sections):

                ref = s["elements"]["A"]

                visual = random.choice(visual_options)

                # evito repetir visual igual seguido (queda más variado)
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

                    # componentes por fase
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
            # 📊 RESULTADOS TRIFÁSICOS
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
                    "P": clean_float(r["P"]),
                    "Q": clean_float(r["Q"]),
                    "S": clean_float(r["S"])
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


        # =====================================================
        # 🔌 CASO MONOFÁSICO (RESTO DE BLOQUES)
        # =====================================================
        circuit = Circuit(rows=rows, cols=cols)
        circuit.solve()

        nodos = []

        # convierto nodos del grafo a JSON
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
                "potential": safe_value(
                    circuit.G.nodes[node].get("potential", 0)
                )
            })

        componentes = []

        # convierto edges (componentes del circuito)
        for i, (u, v) in enumerate(circuit.G.edges()):

            data = circuit.G[u][v]

            componentes.append({
                    "id": f"c{i}",
                    "source": u,
                    "target": v,

                    "type": normalize_type(data.get("element")),
                    "value": str(data.get("string", "")),

                    "orientation": "horizontal" if u[1] == v[1] else "vertical",



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

        # si algo explota, lo enseño en consola y devuelvo error limpio
        print(traceback.format_exc())

        return Response({
            "success": False,
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)