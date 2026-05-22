from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
import random
import random as python_random  # <-- Le damos un alias único para que nadie lo pise
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
        if bloque_id == 4:
            nodos = []
            componentes = []
            pregunta = {}

            # Seleccionamos una plantilla aleatoria (o fija según tu lógica de test)
            plantilla = int(request.data.get('plantilla', 1))

            # -----------------------------------------------------------------
            # 📐 PLANTILLA 1: Método de Nudos (Circuito de la Imagen)
            # -----------------------------------------------------------------
            # -----------------------------------------------------------------
# 📐 PLANTILLA 1: Método de Nudos (Circuito de la Imagen)
# -----------------------------------------------------------------
            if plantilla == 1:
                enunciado_global = (
                    "Determinar las tensiones de los nudos A y B usando el método de los nudos, "
                    "así como la corriente o potencias asociadas a cada elemento de la red."
                )

                # 📌 Asegúrate de que las listas [...] estén bien metidas dentro de los paréntesis (...)
                val_Ig = float(python_random.choice([1, 2, 3, 4, 5]))
                val_R1 = float(python_random.choice([10, 20, 30, 40]))
                val_R2 = float(python_random.choice([5, 15, 25, 35]))
                val_R3 = float(python_random.choice([50, 100, 150]))

                # Resolución analítica exacta por Kirchhoff (Método de Nudos)
                va_solucion = float(val_Ig * (val_R1 + val_R2))
                vb_solucion = 0.0

                # 📌 MAPEADO DE COORDENADAS (Basado fielmente en tu imagen)
                # Fila 0: Cable superior de R1
                # Fila 1: Nudo A y Nudo B con R2 en medio
                # Fila 2: Fuente Ig y Resistencia R3
                # Fila 3: Nudo C de MASA / REFERENCIA
                nodos = [
                    # Esquinas estructurales superiores
                    {"id": "N00", "row": 0, "col": 1, "type": "corner-top-left"},
                    {"id": "N02", "row": 0, "col": 3, "type": "corner-top-right"},

                    # Nudos Principales del Problema
                    {"id": "N10", "row": 1, "col": 1, "type": "center"},          # NUDO A
                    {"id": "N12", "row": 1, "col": 3, "type": "center"},          # NUDO B

                    # Conexiones inferiores hacia Tierra
                    {"id": "N20", "row": 2, "col": 1, "type": "edge-left"},
                    {"id": "N22", "row": 2, "col": 3, "type": "edge-right"},

                    # Línea de Masa (Nudo C)
                    {"id": "N30", "row": 3, "col": 1, "type": "corner-bottom-left"},
                    {"id": "N00_GND", "row": 3, "col": 2, "type": "edge-bottom"}, # Punto exacto de la TIERRA 'N00'
                    {"id": "N32", "row": 3, "col": 3, "type": "corner-bottom-right"}
                ]

                componentes = [
                    # Malla Superior (R1)
                    {"id": "W_A_UP", "source": "N10", "target": "N00", "type": "wire", "value": "", "orientation": "vertical", "labelPosition": "inside-right"},
                    {"id": "R1", "source": "N00", "target": "N02", "type": "resistor", "value": f"{val_R1} Ω", "orientation": "horizontal", "labelPosition": "outside-top"},
                    {"id": "W_B_UP", "source": "N12", "target": "N02", "type": "wire", "value": "", "orientation": "vertical", "labelPosition": "inside-right"},

                    # Rama Central (R2 entre A y B)
                    {"id": "R2", "source": "N10", "target": "N12", "type": "resistor", "value": f"{val_R2} Ω", "orientation": "horizontal", "labelPosition": "outside-top"},

                    # Rama Izquierda: Fuente de Corriente Ig (Va hacia arriba, de N20 a N10)
                    {"id": "Ig", "source": "N20", "target": "N10", "type": "c_source", "value": f"{val_Ig} A", "orientation": "vertical", "labelPosition": "outside-left"},
                    {"id": "W_GND_L", "source": "N30", "target": "N20", "type": "wire", "value": "", "orientation": "vertical", "labelPosition": "inside-right"},

                    # Rama Derecha: Resistencia de caída R3
                    {"id": "R3", "source": "N12", "target": "N22", "type": "resistor", "value": f"{val_R3} Ω", "orientation": "vertical", "labelPosition": "outside-right"},
                    {"id": "W_GND_R", "source": "N32", "target": "N22", "type": "wire", "value": "", "orientation": "vertical", "labelPosition": "inside-right"},

                    # Bus de Tierra Inferior (Nudo C)
                    {"id": "W_C_L", "source": "N30", "target": "N00_GND", "type": "wire", "value": "", "orientation": "horizontal", "labelPosition": "inside-bottom"},
                    {"id": "W_C_R", "source": "N00_GND", "target": "N32", "type": "wire", "value": "", "orientation": "horizontal", "labelPosition": "inside-bottom"}
                ]

                pregunta = {
                    "id": "p1",
                    "enunciado_general": enunciado_global,
                    "datos_enunciado": f"Datos: Ig = {val_Ig} A, R1 = {val_R1} Ω, R2 = {val_R2} Ω, R3 = {val_R3} Ω.",
                    "label": "Tensión en el nudo A (VA)",
                    "unidad": "V",
                    "solucion": round(va_solucion, 2)
                }


            # -------------------------------------------------
            # PLANTILLA 2 (Tensiones de nudos - Problema 2)
            # -------------------------------------------------
            elif plantilla == 2:
                    enunciado_global = (
                        "Determinar las tensiones de los nudos en el circuito de la figura "
                        "y la intensidad por R2 usando el método de los nudos."
                    )

                    # 📌 Valores aleatorios corregidos metiendo las listas dentro de los corchetes [...]
                    val_Ig = float(python_random.choice([1.0, 2.0, 3.0, 4.0, 5.0]))
                    val_R1 = float(python_random.choice([10.0, 20.0, 30.0, 40.0]))
                    val_R2 = float(python_random.choice([5.0, 15.0, 25.0, 35.0]))
                    val_R3 = float(python_random.choice([50.0, 100.0, 150.0]))
                    val_R4 = float(python_random.choice([50.0, 100.0, 150.0]))
                    val_R5 = float(python_random.choice([50.0, 100.0, 150.0]))

                    # 📌 RESOLUCIÓN ANALÍTICA (Sistema de ecuaciones de nudos)
                    import numpy as np

                    g1, g2, g3, g4, g5 = 1/val_R1, 1/val_R2, 1/val_R3, 1/val_R4, 1/val_R5
                    A_matriz = np.array([
                        [g1 + g2,    -g1,         -g2],
                        [-g1,         g1+g3+g4,   -g3],
                        [-g2,        -g3,          g2+g3+g5]
                    ])
                    B_matriz = np.array([val_Ig, 0, 0])

                    try:
                        soluciones = np.linalg.solve(A_matriz, B_matriz)

                        # 📌 Corregido: Extracción escalar usando índices para evitar el TypeError
                        va_sol = float(soluciones[0])
                        vb_sol = float(soluciones[1])
                        vc_sol = float(soluciones[2])

                        # Intensidad por R2 (de A hacia C)
                        i_r2_sol = float((va_sol - vc_sol) / val_R2)
                    except np.linalg.LinAlgError:
                        va_sol, vb_sol, vc_sol, i_r2_sol = 0.0, 0.0, 0.0, 0.0

                    # 📌 MAPEO DE COORDENADAS RECTANGULARES (Malla de 4x4)
                    nodos = [
                        # Esquinas superiores para el bypass de R2
                        {"id": "NW", "row": 0, "col": 1, "type": "corner"},
                        {"id": "NE", "row": 0, "col": 5, "type": "corner"},

                        # Nudos Principales (Fila 1)
                        {"id": "A", "row": 1, "col": 1, "type": "center"},
                        {"id": "B", "row": 1, "col": 3, "type": "center"},
                        {"id": "C", "row": 1, "col": 5, "type": "center"},

                        # Esquinas Inferiores y punto de conexión a Tierra
                        {"id": "SW", "row": 3, "col": 1, "type": "corner"},
                        {"id": "GND", "row": 3, "col": 3, "type": "ground"},
                        {"id": "SE", "row": 3, "col": 5, "type": "corner"}
                    ]

                    componentes = [
                        # --- CAPA SUPERIOR (R2) ---
                        {"id": "W_A_UP", "source": "A", "target": "NW", "type": "wire"},
                        {"id": "R2", "source": "NW", "target": "NE", "type": "resistor", "value": f"{val_R2} Ω", "orientation": "horizontal"},
                        {"id": "W_C_UP", "source": "C", "target": "NE", "type": "wire"},

                        # --- CAPA CENTRAL (R1 y R3) ---
                        {"id": "R1", "source": "A", "target": "B", "type": "resistor", "value": f"{val_R1} Ω", "orientation": "horizontal"},
                        {"id": "R3", "source": "B", "target": "C", "type": "resistor", "value": f"{val_R3} Ω", "orientation": "horizontal"},

                        # --- CAPA VERTICAL (Ig, R4, R5) ---
                        {"id": "Ig", "source": "SW", "target": "A", "type": "c_source", "value": f"{val_Ig} A", "orientation": "vertical"},
                        {"id": "R4", "source": "GND", "target": "B", "type": "resistor", "value": f"{val_R4} Ω", "orientation": "vertical"},
                        {"id": "R5", "source": "SE", "target": "C", "type": "resistor", "value": f"{val_R5} Ω", "orientation": "vertical"},

                        # --- CAPA INFERIOR (Cierre de masa hacia el nodo central GND) ---
                        {"id": "W_GND_L", "source": "SW", "target": "GND", "type": "wire"},
                        {"id": "W_GND_R", "source": "SE", "target": "GND", "type": "wire"}
                    ]

                    pregunta = {
                        "id": "p2",
                        "enunciado_general": enunciado_global,
                        "datos_enunciado": f"Datos: Ig = {val_Ig} A, R1 = {val_R1} Ω, R2 = {val_R2} Ω, R3 = {val_R3} Ω, R4 = {val_R4} Ω, R5 = {val_R5} Ω.",
                        "label": "Intensidad de corriente a través de R2 (I_R2)",
                        "unidad": "A",
                        "solucion": round(i_r2_sol, 3)
                    }

            # -------------------------------------------------
            # PLANTILLA 3 (Tensiones de nudos - Problema 3)
            # -------------------------------------------------
            elif plantilla == 3:
                    enunciado_global = (
                        "Determinar las tensiones de los nudos A, B y C usando el método de los nudos, "
                        "así como las corrientes o potencias asociadas en presencia de la fuente de tensión Eg."
                    )

                    # 📌 Valores aleatorios para los componentes (Incluyendo Eg y Rg)
                    val_Eg = float(python_random.choice([10.0, 12.0, 24.0, 30.0])) # Fuente de tensión en Voltios
                    val_Rg = float(python_random.choice([2.0, 4.0, 5.0]))          # Resistencia de la fuente
                    val_R1 = float(python_random.choice([10.0, 20.0, 30.0]))
                    val_R2 = float(python_random.choice([5.0, 15.0, 25.0]))
                    val_R3 = float(python_random.choice([50.0, 100.0, 150.0]))
                    val_R4 = float(python_random.choice([50.0, 100.0, 150.0]))
                    val_R5 = float(python_random.choice([50.0, 100.0, 150.0]))

                    # 📌 RESOLUCIÓN ANALÍTICA POR MATRICES
                    # Nota: La rama izquierda ahora tiene Eg y Rg en serie conectadas hacia el Nudo A.
                    # La admitancia de esa rama es g_g = 1/Rg. Aporta a la ecuación: (VA - Eg) * g_g -> VA*g_g = Eg*g_g
                    import numpy as np

                    gg, g1, g2, g3, g4, g5 = 1/val_Rg, 1/val_R1, 1/val_R2, 1/val_R3, 1/val_R4, 1/val_R5

                    A_matriz = np.array([
                        [gg + g1 + g2,  -g1,         -g2],
                        [-g1,           g1+g3+g4,    -g3],
                        [-g2,           -g3,          g2+g3+g5]
                    ])
                    # La fuente Eg inyecta corriente al nudo A a través de Rg
                    B_matriz = np.array([val_Eg * gg, 0, 0])

                    try:
                        soluciones = np.linalg.solve(A_matriz, B_matriz)
                        va_sol = float(soluciones[0])
                        vb_sol = float(soluciones[1])
                        vc_sol = float(soluciones[2])
                    except np.linalg.LinAlgError:
                        va_sol, vb_sol, vc_sol = 0.0, 0.0, 0.0

                    # 📌 MAPEO DE COORDENADAS RECTANGULARES (Malla adaptada para añadir Rg)
                    nodos = [
                        # Esquinas superiores para el bypass de R2
                        {"id": "NW", "row": 0, "col": 1, "type": "corner"},
                        {"id": "NE", "row": 0, "col": 5, "type": "corner"},

                        # Nudos Principales (Fila 1)
                        {"id": "A", "row": 1, "col": 1, "type": "center"},
                        {"id": "B", "row": 1, "col": 3, "type": "center"},
                        {"id": "C", "row": 1, "col": 5, "type": "center"},

                        # Puntos de la malla inferior
                        {"id": "SW_MID", "row": 2, "col": 1, "type": "corner"}, # Entre Eg y Rg
                        {"id": "SW",     "row": 3, "col": 1, "type": "corner"}, # Esquina inferior izquierda
                        {"id": "GND",    "row": 3, "col": 3, "type": "ground"}, # Tierra central
                        {"id": "SE",     "row": 3, "col": 5, "type": "corner"}  # Esquina inferior derecha
                    ]

                    componentes = [
                        # --- CAPA SUPERIOR (R2) ---
                        {"id": "W_A_UP", "source": "A", "target": "NW", "type": "wire"},
                        {"id": "R2", "source": "NW", "target": "NE", "type": "resistor", "value": f"{val_R2} Ω", "orientation": "horizontal"},
                        {"id": "W_C_UP", "source": "C", "target": "NE", "type": "wire"},

                        # --- CAPA CENTRAL (R1 y R3) ---
                        {"id": "R1", "source": "A", "target": "B", "type": "resistor", "value": f"{val_R1} Ω", "orientation": "horizontal"},
                        {"id": "R3", "source": "B", "target": "C", "type": "resistor", "value": f"{val_R3} Ω", "orientation": "horizontal"},

                        # --- RAMA IZQUIERDA: Eg y Rg en serie ---
                        # Fuente de tensión Eg vertical subiendo hacia el punto intermedio
                        {"id": "Eg", "source": "SW", "target": "SW_MID", "type": "v_source", "value": f"{val_Eg} V", "orientation": "vertical"},
                        # Cable o continuación vertical directa hacia el nudo A
                        {"id": "W_A_LO", "source": "SW_MID", "target": "A", "type": "wire"},

                        # --- CAPA VERTICAL CENTRAL Y DERECHA (R4, R5) ---
                        {"id": "R4", "source": "GND", "target": "B", "type": "resistor", "value": f"{val_R4} Ω", "orientation": "vertical"},
                        {"id": "R5", "source": "SE", "target": "C", "type": "resistor", "value": f"{val_R5} Ω", "orientation": "vertical"},

                        # --- CAPA INFERIOR: Contiene a Rg antes de llegar a la masa común ---
                        # Resistencia Rg colocada de forma horizontal en la base izquierda
                        {"id": "Rg", "source": "SW", "target": "GND", "type": "resistor", "value": f"{val_Rg} Ω", "orientation": "horizontal"},
                        # Cierre de cable derecho normal hacia la toma de tierra
                        {"id": "W_GND_R", "source": "SE", "target": "GND", "type": "wire"}
                    ]

                    pregunta = {
                        "id": "p3",
                        "enunciado_general": enunciado_global,
                        "datos_enunciado": f"Datos: Eg = {val_Eg} V, Rg = {val_Rg} Ω, R1 = {val_R1} Ω, R2 = {val_R2} Ω, R3 = {val_R3} Ω, R4 = {val_R4} Ω, R5 = {val_R5} Ω.",
                        "label": "Tensión en el nudo A (VA)",
                        "unidad": "V",
                        "solucion": round(va_sol, 2)
                    }
            # -------------------------------------------------
            # PLANTILLA 4 (Tensiones de nudos - Problema 4)
            # -------------------------------------------------
            elif plantilla == 4:
                enunciado_global = (
                    "Determinar las tensiones de los nudos A, B y C usando el método de los nudos, "
                    "teniendo en cuenta la existencia del supernudo formado por la fuente Eg."
                )

                # 📌 Valores aleatorios para los componentes
                val_Eg = float(python_random.choice([5.0, 10.0, 12.0, 15.0])) # Fuente de tensión entre A y B
                val_R1 = float(python_random.choice([10.0, 20.0, 30.0]))
                val_R2 = float(python_random.choice([5.0, 15.0, 25.0]))
                val_R3 = float(python_random.choice([20.0, 40.0, 50.0]))
                val_R4 = float(python_random.choice([50.0, 100.0, 150.0]))
                val_R5 = float(python_random.choice([50.0, 100.0, 150.0]))

                # 📌 RESOLUCIÓN ANALÍTICA (Planteamiento con Supernudo A-B)
                # Ecuación del supernudo (A y B juntos):
                # VA*(1/R1 + 1/R2) + VB*(1/R4 + 1/R3) - VC*(1/R2 + 1/R3) = 0
                # Ecuación del Nudo C:
                # -VA*(1/R2) - VB*(1/R3) + VC*(1/R2 + 1/R3 + 1/R5) = 0
                # Ecuación de ligadura de la fuente:
                # VB - VA = Eg  ->  -VA + VB = Eg  (asumiendo el polo positivo en B)
                import numpy as np

                g1, g2, g3, g4, g5 = 1/val_R1, 1/val_R2, 1/val_R3, 1/val_R4, 1/val_R5

                A_matriz = np.array([
                    [g1 + g2,    g4 + g3,    -(g2 + g3)],  # KCL en el Supernudo A-B
                    [-g2,        -g3,         g2 + g3 + g5], # KCL en el Nudo C
                    [-1.0,        1.0,         0.0]         # Ecuación de ligadura (VB - VA = Eg)
                ])
                B_matriz = np.array([0.0, 0.0, val_Eg])

                try:
                    soluciones = np.linalg.solve(A_matriz, B_matriz)
                    va_sol = float(soluciones[0])
                    vb_sol = float(soluciones[1])
                    vc_sol = float(soluciones[2])
                except np.linalg.LinAlgError:
                    va_sol, vb_sol, vc_sol = 0.0, 0.0, 0.0

                # 📌 MAPEO DE COORDENADAS RECTANGULARES (Malla de 4x5)
                nodos = [
                    # Esquinas superiores para el bypass de R2
                    {"id": "NW", "row": 0, "col": 1, "type": "corner"},
                    {"id": "NE", "row": 0, "col": 5, "type": "corner"},

                    # Nudos Principales (Fila 1)
                    {"id": "A", "row": 1, "col": 1, "type": "center"},
                    {"id": "B", "row": 1, "col": 3, "type": "center"},
                    {"id": "C", "row": 1, "col": 5, "type": "center"},

                    # Capa Inferior
                    {"id": "SW",     "row": 3, "col": 1, "type": "corner"}, # Esquina inferior izquierda
                    {"id": "GND",    "row": 3, "col": 3, "type": "ground"}, # Tierra central
                    {"id": "SE",     "row": 3, "col": 5, "type": "corner"}  # Esquina inferior derecha
                ]

                componentes = [
                    # --- CAPA SUPERIOR (Bypass R2 de A a C) ---
                    {"id": "W_A_UP", "source": "A", "target": "NW", "type": "wire"},
                    {"id": "R2", "source": "NW", "target": "NE", "type": "resistor", "value": f"{val_R2} Ω", "orientation": "horizontal"},
                    {"id": "W_C_UP", "source": "C", "target": "NE", "type": "wire"},

                    # --- CAPA CENTRAL (Fuente Eg entre A-B y Resistencia R3 entre B-C) ---
                    {"id": "Eg", "source": "A", "target": "B", "type": "v_source", "value": f"{val_Eg} V", "orientation": "horizontal"},
                    {"id": "R3", "source": "B", "target": "C", "type": "resistor", "value": f"{val_R3} Ω", "orientation": "horizontal"},

                    # --- CAPA VERTICAL (Líneas de bajada hacia la base) ---
                    {"id": "W_A_LO", "source": "A", "target": "SW", "type": "wire"},
                    {"id": "R4", "source": "GND", "target": "B", "type": "resistor", "value": f"{val_R4} Ω", "orientation": "vertical"},
                    {"id": "R5", "source": "SE", "target": "C", "type": "resistor", "value": f"{val_R5} Ω", "orientation": "vertical"},

                    # --- CAPA INFERIOR (Contiene a R1 y los cierres de masa) ---
                    {"id": "R1", "source": "SW", "target": "GND", "type": "resistor", "value": f"{val_R1} Ω", "orientation": "horizontal"},
                    {"id": "W_GND_R", "source": "SE", "target": "GND", "type": "wire"}
                ]

                pregunta = {
                    "id": "p4",
                    "enunciado_general": enunciado_global,
                    "datos_enunciado": f"Datos: Eg = {val_Eg} V, R1 = {val_R1} Ω, R2 = {val_R2} Ω, R3 = {val_R3} Ω, R4 = {val_R4} Ω, R5 = {val_R5} Ω.",
                    "label": "Tensión en el nudo B (VB)",
                    "unidad": "V",
                    "solucion": round(vb_sol, 2)
                }

            # -------------------------------------------------
            # PLANTILLA 5 (Tensiones de nudos - Problema 5)
            # -------------------------------------------------
            elif plantilla == 5:
                enunciado_global = (
                    "Determinar las tensiones de los nudos A, B y C usando el método de los nudos, "
                    "conociendo la acción conjunta de la fuente de tensión Eg y la fuente de corriente Ig."
                )

                # 📌 Valores aleatorios para los componentes
                val_Eg = float(python_random.choice([10.0, 12.0, 15.0, 24.0])) # En Voltios
                val_Ig = float(python_random.choice([1.0, 2.0, 3.0, 4.0]))     # En Amperios
                val_R1 = float(python_random.choice([10.0, 20.0, 30.0]))
                val_R2 = float(python_random.choice([5.0, 15.0, 25.0]))
                val_R3 = float(python_random.choice([20.0, 40.0, 50.0]))
                val_R4 = float(python_random.choice([50.0, 100.0, 150.0]))
                val_R5 = float(python_random.choice([50.0, 100.0, 150.0]))

                # 📌 RESOLUCIÓN ANALÍTICA (Matriz de nudos reducida ya que VA = Eg)
                # Como VA ya se conoce (Eg), resolvemos para VB y VC:
                # Nudo B: VB * (1/R1 + 1/R3 + 1/R4) - VC * (1/R3) = Eg * (1/R1)
                # Nudo C: -VB * (1/R3) + VC * (1/R2 + 1/R3 + 1/R5) = Eg * (1/R2) + Ig
                import numpy as np

                g1, g2, g3, g4, g5 = 1/val_R1, 1/val_R2, 1/val_R3, 1/val_R4, 1/val_R5

                A_matriz = np.array([
                    [g1 + g3 + g4,   -g3],
                    [-g3,             g2 + g3 + g5]
                ])
                B_matriz = np.array([
                    val_Eg * g1,
                    (val_Eg * g2) + val_Ig
                ])

                try:
                    soluciones = np.linalg.solve(A_matriz, B_matriz)
                    va_sol = float(val_Eg)
                    vb_sol = float(soluciones[0])
                    vc_sol = float(soluciones[1])
                except np.linalg.LinAlgError:
                    va_sol, vb_sol, vc_sol = 0.0, 0.0, 0.0

                # 📌 MAPEO DE COORDENADAS RECTANGULARES (Extendida a columna 6 para la nueva rama)
                nodos = [
                    # Esquinas superiores para el bypass de R2
                    {"id": "NW", "row": 0, "col": 1, "type": "corner"},
                    {"id": "NE", "row": 0, "col": 5, "type": "corner"},

                    # Nudos Principales (Fila 1)
                    {"id": "A", "row": 1, "col": 1, "type": "center"},
                    {"id": "B", "row": 1, "col": 3, "type": "center"},
                    {"id": "C", "row": 1, "col": 5, "type": "center"},

                    # Capa Inferior y Extremos derechos
                    {"id": "SW",     "row": 3, "col": 1, "type": "corner"}, # Abajo de la fuente Eg
                    {"id": "GND",    "row": 3, "col": 3, "type": "ground"}, # Tierra central
                    {"id": "SE",     "row": 3, "col": 5, "type": "corner"}, # Abajo de R5
                    {"id": "SE_EXT", "row": 3, "col": 6, "type": "corner"}  # Abajo de la fuente Ig
                ]

                componentes = [
                    # --- CAPA SUPERIOR (Bypass R2 de A a C) ---
                    {"id": "W_A_UP", "source": "A", "target": "NW", "type": "wire"},
                    {"id": "R2", "source": "NW", "target": "NE", "type": "resistor", "value": f"{val_R2} Ω", "orientation": "horizontal"},
                    {"id": "W_C_UP", "source": "C", "target": "NE", "type": "wire"},

                    # --- CAPA CENTRAL (Resistencias horizontales) ---
                    {"id": "R1", "source": "A", "target": "B", "type": "resistor", "value": f"{val_R1} Ω", "orientation": "horizontal"},
                    {"id": "R3", "source": "B", "target": "C", "type": "resistor", "value": f"{val_R3} Ω", "orientation": "horizontal"},

                    # --- CAPA VERTICAL (Fuentes y resistencias de bajada) ---
                    {"id": "Eg", "source": "SW", "target": "A", "type": "v_source", "value": f"{val_Eg} V", "orientation": "vertical"},
                    {"id": "R4", "source": "GND", "target": "B", "type": "resistor", "value": f"{val_R4} Ω", "orientation": "vertical"},
                    {"id": "R5", "source": "SE", "target": "C", "type": "resistor", "value": f"{val_R5} Ω", "orientation": "vertical"},
                    {"id": "Ig", "source": "SE_EXT", "target": "NE", "type": "c_source", "value": f"{val_Ig} A", "orientation": "vertical"},

                    # --- CAPA INFERIOR (Buses de interconexión a Masa común) ---
                    {"id": "W_GND_L", "source": "SW", "target": "GND", "type": "wire"},
                    {"id": "W_GND_R", "source": "SE", "target": "GND", "type": "wire"},
                    {"id": "W_GND_EXT", "source": "SE", "target": "SE_EXT", "type": "wire"}
                ]

                pregunta = {
                    "id": "p5",
                    "enunciado_general": enunciado_global,
                    "datos_enunciado": f"Datos: Eg = {val_Eg} V, Ig = {val_Ig} A, R1 = {val_R1} Ω, R2 = {val_R2} Ω, R3 = {val_R3} Ω, R4 = {val_R4} Ω, R5 = {val_R5} Ω.",
                    "label": "Tensión en el nudo C (VC)",
                    "unidad": "V",
                    "solucion": round(vc_sol, 2)
                }

            # -------------------------------------------------
            # PLANTILLA 6 (Intensidades de malla - Problema 6)
            # -------------------------------------------------
            elif plantilla == 6:
                enunciado_global = (
                    "Para el circuito de corriente alterna de la figura, determinar las corrientes de malla "
                    "Ia e Ib utilizando el método de las mallas."
                )

                # 📌 Valores aleatorios (Mantenemos magnitudes reales en el choice)
                val_Eg_mag = float(python_random.choice([10.0, 20.0, 50.0, 100.0])) # Magnitud en Voltios (RMS)
                val_R1 = float(python_random.choice([10.0, 20.0, 30.0]))
                val_R2 = float(python_random.choice([5.0, 15.0, 25.0]))
                val_R3 = float(python_random.choice([40.0, 50.0, 60.0]))

                # 📌 RESOLUCIÓN ANALÍTICA (Sistema de mallas en CA)
                # Definimos la fuente como un número complejo puro (fase 0°)
                Eg = complex(val_Eg_mag, 0.0)

                # Planteamiento de ecuaciones de malla:
                # Malla a: Ia * (R1 + R2) - Ib * R2 = 0
                # Malla b: -Ia * R2 + Ib * (R2 + R3) = Eg
                import numpy as np

                Z_matriz = np.array([
                    [complex(val_R1 + val_R2, 0.0),  complex(-val_R2, 0.0)],
                    [complex(-val_R2, 0.0),         complex(val_R2 + val_R3, 0.0)]
                ])
                V_matriz = np.array([complex(0.0, 0.0), Eg])

                try:
                    soluciones = np.linalg.solve(Z_matriz, V_matriz)
                    # Extraemos las corrientes complejas de malla
                    ia_complex = soluciones[0]
                    ib_complex = soluciones[1]

                    # Calculamos la magnitud de la corriente Ib (por ejemplo, para la solución)
                    solucion_magnitud = float(np.abs(ib_complex))
                except np.linalg.LinAlgError:
                    solucion_magnitud = 0.0

                # 📌 MAPEO DE COORDENADAS RECTANGULARES (Estructura de doble malla vertical)
                nodos = [
                    {"id": "NW", "row": 0, "col": 1, "type": "corner"},
                    {"id": "NE", "row": 0, "col": 3, "type": "corner"},
                    {"id": "A",  "row": 1, "col": 1, "type": "node"},    # Conexión intermedia izquierda
                    {"id": "B",  "row": 1, "col": 3, "type": "node"},    # Conexión intermedia derecha
                    {"id": "SW", "row": 2, "col": 1, "type": "corner"},
                    {"id": "SE", "row": 2, "col": 3, "type": "corner"}
                ]

                componentes = [
                    # --- MALLA SUPERIOR (Malla a) ---
                    {"id": "R1", "source": "NW", "target": "NE", "type": "resistor", "value": f"{val_R1} Ω", "orientation": "horizontal"},
                    {"id": "W_L_UP", "source": "NW", "target": "A", "type": "wire"},
                    {"id": "W_R_UP", "source": "NE", "target": "B", "type": "wire"},

                    # --- DIVISOR CENTRAL (Compartido entre Malla a y b) ---
                    {"id": "R2", "source": "A", "target": "B", "type": "resistor", "value": f"{val_R2} Ω", "orientation": "horizontal"},

                    # --- MALLA INFERIOR (Malla b) ---
                    {"id": "Eg", "source": "SW", "target": "A", "type": "ac_source", "value": f"{val_Eg_mag} V", "orientation": "vertical"},
                    {"id": "R3", "source": "SE", "target": "B", "type": "resistor", "value": f"{val_R3} Ω", "orientation": "vertical"},
                    {"id": "W_LO_BOT", "source": "SW", "target": "SE", "type": "wire"}
                ]

                pregunta = {
                    "id": "p6",
                    "enunciado_general": enunciado_global,
                    "datos_enunciado": f"Datos: Eg = {val_Eg_mag} V (CA), R1 = {val_R1} Ω, R2 = {val_R2} Ω, R3 = {val_R3} Ω.",
                    "label": "Magnitud de la corriente de malla Ib",
                    "unidad": "A",
                    "solucion": round(solucion_magnitud, 2)
                }

            # -------------------------------------------------
            # PLANTILLA 7 (Intensidades de malla - Problema 7)
            # -------------------------------------------------
            elif plantilla == 7:
                    enunciado_global = (
                        "Determinar las corrientes de malla Ia, Ib e Ic en el circuito de la figura "
                        "utilizando el método de las mallas."
                    )

                    # 📌 Valores aleatorios para los componentes
                    val_Ig = float(python_random.choice([1.0, 2.0, 3.0, 4.0]))     # Fuente de corriente (hacia abajo)
                    val_Rg = float(python_random.choice([10.0, 20.0, 30.0]))
                    val_R1 = float(python_random.choice([10.0, 15.0, 25.0]))
                    val_R2 = float(python_random.choice([5.0, 10.0, 20.0]))
                    val_R3 = float(python_random.choice([20.0, 40.0, 50.0]))
                    val_R4 = float(python_random.choice([50.0, 100.0, 150.0]))
                    val_R5 = float(python_random.choice([50.0, 100.0, 150.0]))

                    # 📌 RESOLUCIÓN ANALÍTICA (Sistema de 3 mallas: Ia, Ib, Ic)
                    # Nota: La fuente de corriente Ig externa define una relación, pero al estar en paralelo con Rg,
                    # podemos plantear las 3 mallas tradicionales internas de las ventanas:
                    # Malla a (inferior izquierda): Ia * (Rg + R1 + R4) - Ib * R1 - Ic * R4 = -Ig * Rg
                    # Malla b (superior):         -Ia * R1 + Ib * (R1 + R2 + R3) - Ic * R3 = 0
                    # Malla c (inferior derecha):   -Ia * R4 - Ib * R3 + Ic * (R3 + R4 + R5) = 0
                    import numpy as np

                    A_matriz = np.array([
                        [val_Rg + val_R1 + val_R4,  -val_R1,                    -val_R4],
                        [-val_R1,                    val_R1 + val_R2 + val_R3,  -val_R3],
                        [-val_R4,                   -val_R3,                     val_R3 + val_R4 + val_R5]
                    ])
                    # -Ig * Rg porque Ig va hacia abajo e ingresa en sentido opuesto a Ia en la rama externa
                    B_matriz = np.array([-val_Ig * val_Rg, 0.0, 0.0])

                    try:
                        soluciones = np.linalg.solve(A_matriz, B_matriz)
                        ia_sol = float(soluciones[0])
                        ib_sol = float(soluciones[1])
                        ic_sol = float(soluciones[2])
                    except np.linalg.LinAlgError:
                        ia_sol, ib_sol, ic_sol = 0.0, 0.0, 0.0

                    # 📌 MAPEO DE COORDENADAS RECTANGULARES (Malla de 4x6 para albergar la rama externa de Ig)
                    nodos = [
                        # Esquinas de la ventana superior (Malla b)
                        {"id": "NW_SUP", "row": 0, "col": 2, "type": "corner"},
                        {"id": "NE_SUP", "row": 0, "col": 5, "type": "corner"},

                        # Fila intermedia (Nudos de distribución)
                        {"id": "W_EXT",  "row": 1, "col": 1, "type": "corner"}, # Esquina de la fuente Ig
                        {"id": "A",      "row": 1, "col": 2, "type": "node"},   # Nudo izquierdo del puente
                        {"id": "B",      "row": 1, "col": 4, "type": "node"},   # Nudo central del puente
                        {"id": "C",      "row": 1, "col": 5, "type": "node"},   # Nudo derecho del puente

                        # Fila inferior (Línea de base común)
                        {"id": "SW_EXT", "row": 3, "col": 1, "type": "corner"},
                        {"id": "SW",     "row": 3, "col": 2, "type": "corner"},
                        {"id": "GND",    "row": 3, "col": 4, "type": "corner"}, # Nodo inferior central
                        {"id": "SE",     "row": 3, "col": 5, "type": "corner"}
                    ]

                    componentes = [
                        # --- RAMA EXTREMA IZQUIERDA (Fuente de corriente Ig) ---
                        {"id": "Ig", "source": "W_EXT", "target": "SW_EXT", "type": "c_source", "value": f"{val_Ig} A", "orientation": "vertical"},
                        {"id": "W_LT_UP", "source": "W_EXT", "target": "A", "type": "wire"},
                        {"id": "W_LT_LO", "source": "SW_EXT", "target": "SW", "type": "wire"},

                        # --- RAMA EN PARALELO Rg ---
                        {"id": "Rg", "source": "SW", "target": "A", "type": "resistor", "value": f"{val_Rg} Ω", "orientation": "vertical"},

                        # --- VENTANA SUPERIOR (R2) ---
                        {"id": "W_A_UP", "source": "A", "target": "NW_SUP", "type": "wire"},
                        {"id": "R2", "source": "NW_SUP", "target": "NE_SUP", "type": "resistor", "value": f"{val_R2} Ω", "orientation": "horizontal"},
                        {"id": "W_C_UP", "source": "C", "target": "NE_SUP", "type": "wire"},

                        # --- LINEA MEDIA DEL PUENTE (R1 y R3) ---
                        {"id": "R1", "source": "A", "target": "B", "type": "resistor", "value": f"{val_R1} Ω", "orientation": "horizontal"},
                        {"id": "R3", "source": "B", "target": "C", "type": "resistor", "value": f"{val_R3} Ω", "orientation": "horizontal"},

                        # --- RAMAS VERTICALES INFERIORES (R4 y R5) ---
                        {"id": "R4", "source": "GND", "target": "B", "type": "resistor", "value": f"{val_R4} Ω", "orientation": "vertical"},
                        {"id": "R5", "source": "SE", "target": "C", "type": "resistor", "value": f"{val_R5} Ω", "orientation": "vertical"},

                        # --- LÍNEA INFERIOR DE CIERRE ---
                        {"id": "W_LO_1", "source": "SW", "target": "GND", "type": "wire"},
                        {"id": "W_LO_2", "source": "SE", "target": "GND", "type": "wire"}
                    ]

                    pregunta = {
                        "id": "p7",
                        "enunciado_general": enunciado_global,
                        "datos_enunciado": f"Datos: Ig = {val_Ig} A, Rg = {val_Rg} Ω, R1 = {val_R1} Ω, R2 = {val_R2} Ω, R3 = {val_R3} Ω, R4 = {val_R4} Ω, R5 = {val_R5} Ω.",
                        "label": "Corriente de malla Ia",
                        "unidad": "A",
                        "solucion": round(ia_sol, 2)
                    }
            # -------------------------------------------------
            # PLANTILLA 8 (Intensidades de malla - Problema 8)
            # -------------------------------------------------
            elif plantilla == 8:
                    enunciado_global = (
                        "Determinar las corrientes de malla Ia, Ib e Ic utilizando el método de mallas, "
                        "considerando la presencia de la supermalla generada por la fuente de corriente interna Ig."
                    )

                    # 📌 Valores aleatorios para los componentes
                    val_Ig = float(python_random.choice([1.0, 1.5, 2.0, 2.5])) # Fuente de corriente intermedia
                    val_R1 = float(python_random.choice([10.0, 20.0, 30.0]))
                    val_R2 = float(python_random.choice([15.0, 25.0, 35.0]))
                    val_R3 = float(python_random.choice([10.0, 20.0, 40.0]))
                    val_R4 = float(python_random.choice([50.0, 100.0, 150.0]))
                    val_R5 = float(python_random.choice([50.0, 100.0, 150.0]))

                    # 📌 RESOLUCIÓN ANALÍTICA (Planteamiento con Supermalla)
                    # Ecuación de Malla a: Ia * (R1 + R2 + R4) - Ib * R2 - Ic * R4 = 0
                    # Ecuación de la Supermalla (b + c): -Ia * R2 + Ib * R2 + Ib * R3 - Ia * R4 + Ic * R4 + Ic * R5 = 0
                    # Simplificando la Supermalla: Ia * (-R2 - R4) + Ib * (R2 + R3) + Ic * (R4 + R5) = 0
                    # Ecuación de ligadura de la fuente Ig (apunta a la derecha: de la malla c a la malla b en la frontera):
                    # La corriente Ig va a favor de Ic y en contra de Ib en esa rama central-derecha -> Ic - Ib = Ig  -> -Ib + Ic = Ig
                    import numpy as np

                    A_matriz = np.array([
                        [val_R1 + val_R2 + val_R4,   -val_R2,                    -val_R4],
                        [-(val_R2 + val_R4),          val_R2 + val_R3,            val_R4 + val_R5],
                        [0.0,                        -1.0,                        1.0]
                    ])
                    B_matriz = np.array([0.0, 0.0, val_Ig])

                    try:
                        soluciones = np.linalg.solve(A_matriz, B_matriz)
                        ia_sol = float(soluciones[0])
                        ib_sol = float(soluciones[1])
                        ic_sol = float(soluciones[2])
                    except np.linalg.LinAlgError:
                        ia_sol, ib_sol, ic_sol = 0.0, 0.0, 0.0

                    # 📌 MAPEO DE COORDENADAS RECTANGULARES (Malla de 4x5)
                    nodos = [
                        # Esquinas Superiores (Malla b)
                        {"id": "NW_SUP", "row": 0, "col": 1, "type": "corner"},
                        {"id": "NE_SUP", "row": 0, "col": 5, "type": "corner"},

                        # Nudos de la línea media
                        {"id": "A",      "row": 1, "col": 1, "type": "node"},   # Extremo izquierdo de R2
                        {"id": "B",      "row": 1, "col": 3, "type": "node"},   # Centro (entre R2, R4 e Ig)
                        {"id": "C",      "row": 1, "col": 5, "type": "node"},   # Extremo derecho de Ig y R5

                        # Base inferior del circuito
                        {"id": "SW",     "row": 3, "col": 1, "type": "corner"},
                        {"id": "GND",    "row": 3, "col": 3, "type": "corner"},
                        {"id": "SE",     "row": 3, "col": 5, "type": "corner"}
                    ]

                    componentes = [
                        # --- CAPA SUPERIOR (R3 coronando la Malla b) ---
                        {"id": "W_A_UP", "source": "A", "target": "NW_SUP", "type": "wire"},
                        {"id": "R3", "source": "NW_SUP", "target": "NE_SUP", "type": "resistor", "value": f"{val_R3} Ω", "orientation": "horizontal"},
                        {"id": "W_C_UP", "source": "C", "target": "NE_SUP", "type": "wire"},

                        # --- LÍNEA INTERMEDIA ---
                        {"id": "R2", "source": "A", "target": "B", "type": "resistor", "value": f"{val_R2} Ω", "orientation": "horizontal"},
                        # Fuente Ig horizontal entre el nudo central B y el nudo derecho C
                        {"id": "Ig", "source": "B", "target": "C", "type": "c_source", "value": f"{val_Ig} A", "orientation": "horizontal"},

                        # --- CAPA VERTICAL ---
                        {"id": "R1", "source": "SW", "target": "A", "type": "resistor", "value": f"{val_R1} Ω", "orientation": "vertical"},
                        {"id": "R4", "source": "GND", "target": "B", "type": "resistor", "value": f"{val_R4} Ω", "orientation": "vertical"},
                        {"id": "R5", "source": "SE", "target": "C", "type": "resistor", "value": f"{val_R5} Ω", "orientation": "vertical"},

                        # --- LÍNEA INFERIOR DE CIERRE ---
                        {"id": "W_LO_1", "source": "SW", "target": "GND", "type": "wire"},
                        {"id": "W_LO_2", "source": "SE", "target": "GND", "type": "wire"}
                    ]

                    pregunta = {
                        "id": "p8",
                        "enunciado_general": enunciado_global,
                        "datos_enunciado": f"Datos: Ig = {val_Ig} A, R1 = {val_R1} Ω, R2 = {val_R2} Ω, R3 = {val_R3} Ω, R4 = {val_R4} Ω, R5 = {val_R5} Ω.",
                        "label": "Corriente de malla Ic",
                        "unidad": "A",
                        "solucion": round(ic_sol, 2)
                    }

            # -------------------------------------------------
            # PLANTILLA 9 (Intensidades de malla - Problema 9)
            # -------------------------------------------------
            elif plantilla == 9:
                    enunciado_global = (
                        "Determinar las corrientes de malla Ia, Ib e Ic utilizando el método de mallas, "
                        "aprovechando que la fuente de corriente externa fija de manera directa el valor de Ib."
                    )

                    # 📌 Valores aleatorios para los componentes
                    val_Ig = float(python_random.choice([1.0, 2.0, 3.0]))          # Fuente de corriente superior
                    val_Eg = float(python_random.choice([10.0, 12.0, 20.0, 24.0])) # Fuente de tensión central-derecha
                    val_R1 = float(python_random.choice([10.0, 20.0, 30.0]))
                    val_R2 = float(python_random.choice([15.0, 25.0, 35.0]))
                    val_R3 = float(python_random.choice([10.0, 20.0, 40.0]))
                    val_R4 = float(python_random.choice([50.0, 100.0, 150.0]))
                    val_R5 = float(python_random.choice([50.0, 100.0, 150.0]))

                    # 📌 RESOLUCIÓN ANALÍTICA
                    # Al estar Ig en la periferia de la malla b: Ib = val_Ig
                    # Nos queda un sistema de 2 ecuaciones con 2 incógnitas (Ia, Ic):
                    # Ecuación Malla a: Ia * (R1 + R2 + R4) - Ic * R4 = Ib * R2  ->  Ia * (R1 + R2 + R4) - Ic * R4 = val_Ig * R2
                    # Ecuación Malla c: -Ia * R4 + Ic * (R4 + R5) = -Eg           (Eg se opone al sentido de Ic en esa rama central)
                    import numpy as np

                    A_matriz = np.array([
                        [val_R1 + val_R2 + val_R4,  -val_R4],
                        [-val_R4,                    val_R4 + val_R5]
                    ])
                    B_matriz = np.array([
                        val_Ig * val_R2,
                        -val_Eg
                    ])

                    try:
                        soluciones = np.linalg.solve(A_matriz, B_matriz)
                        ia_sol = float(soluciones[0])
                        ib_sol = float(val_Ig)
                        ic_sol = float(soluciones[1])
                    except np.linalg.LinAlgError:
                        ia_sol, ib_sol, ic_sol = 0.0, 0.0, 0.0

                    # 📌 MAPEO DE COORDENADAS RECTANGULARES (Malla de 4x5)
                    nodos = [
                        # Esquinas Superiores (Malla b)
                        {"id": "NW_SUP", "row": 0, "col": 1, "type": "corner"},
                        {"id": "N_MID",   "row": 0, "col": 3, "type": "corner"}, # Punto medio superior
                        {"id": "NE_SUP", "row": 0, "col": 5, "type": "corner"},

                        # Nudos de la línea media
                        {"id": "A",      "row": 1, "col": 1, "type": "node"},   # Izquierda de R2
                        {"id": "B",      "row": 1, "col": 3, "type": "node"},   # Centro (entre R2, R4 y Eg)
                        {"id": "C",      "row": 1, "col": 5, "type": "node"},   # Derecha de Eg y R5

                        # Base inferior
                        {"id": "SW",     "row": 3, "col": 1, "type": "corner"},
                        {"id": "GND",    "row": 3, "col": 3, "type": "corner"},
                        {"id": "SE",     "row": 3, "col": 5, "type": "corner"}
                    ]

                    componentes = [
                        # --- CAPA SUPERIOR (Fuente Ig en serie con R3) ---
                        {"id": "W_A_UP", "source": "A", "target": "NW_SUP", "type": "wire"},
                        {"id": "Ig", "source": "NW_SUP", "target": "N_MID", "type": "c_source", "value": f"{val_Ig} A", "orientation": "horizontal"},
                        {"id": "R3", "source": "N_MID", "target": "NE_SUP", "type": f"{val_R3} Ω", "orientation": "horizontal"},
                        {"id": "W_C_UP", "source": "C", "target": "NE_SUP", "type": "wire"},

                        # --- LÍNEA INTERMEDIA ---
                        {"id": "R2", "source": "A", "target": "B", "type": "resistor", "value": f"{val_R2} Ω", "orientation": "horizontal"},
                        {"id": "Eg", "source": "B", "target": "C", "type": "v_source", "value": f"{val_Eg} V", "orientation": "horizontal"},

                        # --- CAPA VERTICAL ---
                        {"id": "R1", "source": "SW", "target": "A", "type": "resistor", "value": f"{val_R1} Ω", "orientation": "vertical"},
                        {"id": "R4", "source": "GND", "target": "B", "type": "resistor", "value": f"{val_R4} Ω", "orientation": "vertical"},
                        {"id": "R5", "source": "SE", "target": "C", "type": "resistor", "value": f"{val_R5} Ω", "orientation": "vertical"},

                        # --- LÍNEA INFERIOR DE CIERRE ---
                        {"id": "W_LO_1", "source": "SW", "target": "GND", "type": "wire"},
                        {"id": "W_LO_2", "source": "SE", "target": "GND", "type": "wire"}
                    ]

                    pregunta = {
                        "id": "p9",
                        "enunciado_general": enunciado_global,
                        "datos_enunciado": f"Datos: Ig = {val_Ig} A, Eg = {val_Eg} V, R1 = {val_R1} Ω, R2 = {val_R2} Ω, R3 = {val_R3} Ω, R4 = {val_R4} Ω, R5 = {val_R5} Ω.",
                        "label": "Corriente de malla Ia",
                        "unidad": "A",
                        "solucion": round(ia_sol, 2)
                    }
            # Envoltura final idéntica a tus especificaciones del JSON
            return Response({
                "success": True,
                "tipo": "nudos_mallas",
                "circuito": {
                    "plantilla": plantilla,
                    "rows": 4, # Incrementado para dar más margen y definición al grid 2D
                    "cols": 5,
                    "nodos": nodos,
                    "componentes": componentes,
                    "preguntas": pregunta
                }
            }, status=status.HTTP_200_OK)


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