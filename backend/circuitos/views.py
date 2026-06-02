# Módulos nativos de Python
import random
import random as python_random
import re
import traceback

# Librerías externas (Data science y la API de Django)
import numpy as np
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response


from .alvaro.lib_new import Circuit, ThreePhaseCircuit


# Convierte números complejos a diccionario porque Angular no entiende el tipo 'complex' de Python
def complex_to_dict(z):
    if z is None:
        return {"re": 0.0, "im": 0.0}

    # Usamos float() para curarnos en salud por si viene de NumPy
    return {
        "re": float(np.real(z)),
        "im": float(np.imag(z))
    }

# Normaliza los datos para que el JSON de respuesta no rompa en el frontend
def safe_value(v):
    if v is None:
        return 0.0

    if isinstance(v, (complex, np.complexfloating)):
        return complex_to_dict(v)

    # Si es un float/int raro de NumPy, lo casteamos a tipo nativo de Python
    if isinstance(v, np.generic):
        return float(v)

    # Si es un string, le quitamos los espacios en blanco que sobran
    if isinstance(v, str):
        return v.strip()

    return v


# Convierte cualquier entrada a un float seguro, manejando complejos y redondeos
def safe_float(v):
    if v is None:
        return 0.0

    # Pillamos el módulo si es complejo (añadido np.complexfloating por si acaso)
    if isinstance(v, (complex, np.complexfloating)):
        v = abs(v)

    # Si es un tipo numérico de NumPy, lo pasamos a float nativo
    if isinstance(v, np.number):
        v = float(v)

    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0

    # Limpieza de ruido numérico extremo
    if abs(v) < 1e-12:
        return 0.0

    # Redondeo limpio para que el front no reciba infinitos decimales
    return round(v, 6)

# Normaliza el string del componente para que el frontend sepa qué icono pintar
def normalize_type(t):
    if not t:
        return "unknown"

    t = str(t).lower()

    # Mapeo por palabras clave
    if "res" in t:
        return "resistor"
    if "cap" in t:
        return "capacitor"
    if "ind" in t:
        return "inductor"

    # Fuentes de energía
    if "c_source" in t or "current" in t:
        return "c_source"
    if "v_source" in t or "volt" in t:
        return "v_source"

    if "wire" in t:
        return "wire"

    return t

# Versión rápida de safe_float (fallback por si necesitas menos redondeo)
def clean_float(v, eps=1e-6):
    if v is None:
        return 0.0

    try:
        v = float(v)
        # Si está muy cerca de cero, lo mandamos a cero directo
        return 0.0 if abs(v) < eps else v
    except (TypeError, ValueError):
        return 0.0

# Define el tipo de nodo según su posición para maquetar la cuadrícula en el front
def determinar_tipo_nodo(row, col, rows, cols):
    is_top = row == 0
    is_bottom = row == rows - 1
    is_left = col == 0
    is_right = col == cols - 1

    # Comprobamos las esquinas de la matriz
    if is_top and is_left:
        return "corner-top-left"
    if is_top and is_right:
        return "corner-top-right"
    if is_bottom and is_left:
        return "corner-bottom-left"
    if is_bottom and is_right:
        return "corner-bottom-right"

    # Comprobamos los bordes de la matriz
    if is_top:
        return "edge-top"
    if is_bottom:
        return "edge-bottom"
    if is_left:
        return "edge-left"
    if is_right:
        return "edge-right"

    return "center"


# API para generar circuitos
@api_view(['POST'])
@permission_classes([AllowAny])
def generar_circuito(request):

    try:

        # Datos recibidos desde el frontend
        bloque_id = int(request.data.get('bloque', 1))
        rows = int(request.data.get('rows', 2))
        cols = int(request.data.get('cols', 3))

        # Bloque 4: ejercicios de análisis de nudos y mallas
        if bloque_id == 4:

            nodos = []
            componentes = []
            preguntas = []

            # Plantilla seleccionada por el usuario
            plantilla = int(request.data.get('plantilla', 1))

            # Plantilla 1 - Método de los nudos
            if plantilla == 1:

                # Enunciado del ejercicio
                enunciado_global = (
                    "Determinar las tensiones de los nudos A y B usando el método de los nudos, "
                    "así como la corriente o potencias asociadas a cada elemento de la red."
                )

                # Valores del circuito
                val_Ig = 10.0
                val_R1 = 1.0
                val_R2 = 1.0
                val_R3 = 1.0

                # Resultados esperados del ejercicio
                va_solucion = 15.0
                vb_solucion = 10.0
                pr1_solucion = 25.0
                pig_solucion = 150.0

                # Nodos utilizados para dibujar el circuito
                nodos = [

                    # Parte superior
                    {"id": "N00", "row": 0, "col": 1, "type": "corner-top-left"},
                    {"id": "N02", "row": 0, "col": 3, "type": "corner-top-right"},

                    # Nudos principales
                    {"id": "N10", "row": 1, "col": 1, "type": "center"},  # Nodo A
                    {"id": "N12", "row": 1, "col": 3, "type": "center"},  # Nodo B

                    # Conexiones inferiores
                    {"id": "N20", "row": 2, "col": 1, "type": "edge-left"},
                    {"id": "N22", "row": 2, "col": 3, "type": "edge-right"},

                    # Masa o referencia
                    {"id": "N30", "row": 3, "col": 1, "type": "corner-bottom-left"},
                    {"id": "N00_GND", "row": 3, "col": 2, "type": "edge-bottom"},
                    {"id": "N32", "row": 3, "col": 3, "type": "corner-bottom-right"}
                ]

                # Componentes del circuito
                componentes = [

                    # Rama superior
                    {"id": "W_A_UP", "source": "N10", "target": "N00", "type": "wire", "value": "", "orientation": "vertical", "labelPosition": "inside-right"},
                    {"id": "R1", "source": "N00", "target": "N02", "type": "resistor", "value": f"{val_R1} Ω", "orientation": "horizontal", "labelPosition": "outside-top"},
                    {"id": "W_B_UP", "source": "N12", "target": "N02", "type": "wire", "value": "", "orientation": "vertical", "labelPosition": "inside-right"},

                    # Resistencia entre los nodos A y B
                    {"id": "R2", "source": "N10", "target": "N12", "type": "resistor", "value": f"{val_R2} Ω", "orientation": "horizontal", "labelPosition": "outside-top"},

                    # Rama izquierda
                    {"id": "Ig", "source": "N20", "target": "N10", "type": "c_source", "value": f"{val_Ig} A", "orientation": "vertical", "labelPosition": "outside-left"},
                    {"id": "W_GND_L", "source": "N30", "target": "N20", "type": "wire", "value": "", "orientation": "vertical", "labelPosition": "inside-right"},

                    # Rama derecha
                    {"id": "R3", "source": "N12", "target": "N22", "type": "resistor", "value": f"{val_R3} Ω", "orientation": "vertical", "labelPosition": "outside-right"},
                    {"id": "W_GND_R", "source": "N32", "target": "N22", "type": "wire", "value": "", "orientation": "vertical", "labelPosition": "inside-right"},

                    # Línea de masa
                    {"id": "W_C_L", "source": "N30", "target": "N00_GND", "type": "wire", "value": "", "orientation": "horizontal", "labelPosition": "inside-bottom"},
                    {"id": "W_C_R", "source": "N00_GND", "target": "N32", "type": "wire", "value": "", "orientation": "horizontal", "labelPosition": "inside-bottom"}
                ]

                # Preguntas que verá el alumno
                preguntas = {
                    "id": "p1",
                    "enunciado_general": enunciado_global,
                    "datos_enunciado": "Datos: Ig = 10 A, R1 = R2 = R3 = 1Ω.",

                    "items": [
                        {
                            "label": "Tensión en el nudo A (VA)",
                            "unidad": "V",
                            "solucion": va_solucion
                        },
                        {
                            "label": "Tensión en el nudo B (VB)",
                            "unidad": "V",
                            "solucion": vb_solucion
                        },
                        {
                            "label": "Potencia en R1",
                            "unidad": "W",
                            "solucion": pr1_solucion
                        },
                        {
                            "label": "Potencia de la fuente Ig",
                            "unidad": "W",
                            "solucion": pig_solucion,
                            "nota": "cedida"
                        }
                    ]
                }

           # Plantilla 2 - Tensiones de nudos (problema 2)
            elif plantilla == 2:

                # Enunciado del ejercicio
                enunciado_global = (
                    "Determinar las tensiones de los nudos en el circuito de la figura "
                    "y la intensidad por R2 usando el método de los nudos."
                )

                # Valores del circuito
                val_Ig = 10.0
                val_R1 = 2.0
                val_R2 = 2.0
                val_R3 = 4.0
                val_R4 = 4.0
                val_R5 = 4.0

                # Resolución del sistema de ecuaciones (método de nudos)
                import numpy as np

                g1, g2, g3, g4, g5 = (
                    1/val_R1,
                    1/val_R2,
                    1/val_R3,
                    1/val_R4,
                    1/val_R5
                )

                A_matriz = np.array([
                    [g1 + g2,    -g1,         -g2],
                    [-g1,         g1 + g3 + g4, -g3],
                    [-g2,        -g3,          g2 + g3 + g5]
                ])

                B_matriz = np.array([val_Ig, 0, 0])

                try:
                    # Resolver sistema de ecuaciones
                    soluciones = np.linalg.solve(A_matriz, B_matriz)

                    # Tensiones de los nodos (valores ya definidos en el ejercicio)
                    va_sol = 30.0
                    vb_sol = 20.0
                    vc_sol = 20.0

                    # Corriente por R2 (de A a C)
                    i_r2_sol = float((va_sol - vc_sol) / val_R2)

                except np.linalg.LinAlgError:
                    # Si el sistema no tiene solución
                    va_sol = vb_sol = vc_sol = 0.0
                    i_r2_sol = 0.0

                # Nodos del circuito
                nodos = [
                    {"id": "NW", "row": 0, "col": 1, "type": "corner"},
                    {"id": "NE", "row": 0, "col": 5, "type": "corner"},

                    {"id": "A", "row": 1, "col": 1, "type": "center"},
                    {"id": "B", "row": 1, "col": 3, "type": "center"},
                    {"id": "C", "row": 1, "col": 5, "type": "center"},

                    {"id": "SW", "row": 3, "col": 1, "type": "corner"},
                    {"id": "GND", "row": 3, "col": 3, "type": "ground"},
                    {"id": "SE", "row": 3, "col": 5, "type": "corner"}
                ]

                # Componentes del circuito
                componentes = [
                    {"id": "W_A_UP", "source": "A", "target": "NW", "type": "wire"},
                    {"id": "R2", "source": "NW", "target": "NE", "type": "resistor", "value": f"{val_R2} Ω"},
                    {"id": "W_C_UP", "source": "C", "target": "NE", "type": "wire"},

                    {"id": "R1", "source": "A", "target": "B", "type": "resistor", "value": f"{val_R1} Ω"},
                    {"id": "R3", "source": "B", "target": "C", "type": "resistor", "value": f"{val_R3} Ω"},

                    {"id": "Ig", "source": "SW", "target": "A", "type": "c_source", "value": f"{val_Ig} A"},
                    {"id": "R4", "source": "GND", "target": "B", "type": "resistor", "value": f"{val_R4} Ω"},
                    {"id": "R5", "source": "SE", "target": "C", "type": "resistor", "value": f"{val_R5} Ω"},

                    {"id": "W_GND_L", "source": "SW", "target": "GND", "type": "wire"},
                    {"id": "W_GND_R", "source": "SE", "target": "GND", "type": "wire"}
                ]

                # Preguntas del ejercicio
                preguntas = {
                    "id": "p2",
                    "enunciado_general": enunciado_global,
                    "datos_enunciado": "Datos: Ig = 10A, R1 = 2Ω, R2 = 2Ω, R3 = R4 = R5 = 4Ω",

                    "items": [
                        {"label": "Tensión en A (VA)", "unidad": "V", "solucion": va_sol},
                        {"label": "Tensión en B (VB)", "unidad": "V", "solucion": vb_sol},
                        {"label": "Tensión en C (VC)", "unidad": "V", "solucion": vc_sol},
                        {"label": "Corriente por R2", "unidad": "A", "solucion": i_r2_sol}
                    ]
                }

            # PLANTILLA 3 (Tensiones de nudos - Problema 3)

            elif plantilla == 3:
                enunciado_global = (
                    "Determinar las tensiones de los nudos A, B y C usando el método de los nudos "
                    "con fuente de tensión Eg."
                )

                # Valores del circuito
                val_Eg = 20.0
                val_Rg = 2.0
                val_R1 = 2.0
                val_R2 = 2.0
                val_R3 = 4.0
                val_R4 = 4.0
                val_R5 = 4.0

                import numpy as np

                gg, g1, g2, g3, g4, g5 = 1/val_Rg, 1/val_R1, 1/val_R2, 1/val_R3, 1/val_R4, 1/val_R5

                A_matriz = np.array([
                    [gg + g1 + g2,  -g1,        -g2],
                    [-g1,           g1+g3+g4,   -g3],
                    [-g2,           -g3,        g2+g3+g5]
                ])

                B_matriz = np.array([val_Eg * gg, 0, 0])

                try:
                    soluciones = np.linalg.solve(A_matriz, B_matriz)

                    va_sol = 12.0
                    vb_sol = 8.0
                    vc_sol = 8.0

                    p_total = 80.0

                except np.linalg.LinAlgError:
                    va_sol, vb_sol, vc_sol = 0.0, 0.0, 0.0
                    p_total = 0.0

                nodos = [
                    {"id": "NW", "row": 0, "col": 1, "type": "corner"},
                    {"id": "NE", "row": 0, "col": 5, "type": "corner"},

                    {"id": "A", "row": 1, "col": 1, "type": "center"},
                    {"id": "B", "row": 1, "col": 3, "type": "center"},
                    {"id": "C", "row": 1, "col": 5, "type": "center"},

                    {"id": "SW_MID", "row": 2, "col": 1, "type": "corner"},
                    {"id": "SW", "row": 3, "col": 1, "type": "corner"},
                    {"id": "GND", "row": 3, "col": 3, "type": "ground"},
                    {"id": "SE", "row": 3, "col": 5, "type": "corner"}
                ]

                componentes = [
                    {"id": "W_A_UP", "source": "A", "target": "NW", "type": "wire"},
                    {"id": "R2", "source": "NW", "target": "NE", "type": "resistor", "value": f"{val_R2} Ω"},
                    {"id": "W_C_UP", "source": "C", "target": "NE", "type": "wire"},

                    {"id": "R1", "source": "A", "target": "B", "type": "resistor", "value": f"{val_R1} Ω"},
                    {"id": "R3", "source": "B", "target": "C", "type": "resistor", "value": f"{val_R3} Ω"},

                    {"id": "Eg", "source": "SW", "target": "SW_MID", "type": "v_source", "value": f"{val_Eg} V"},
                    {"id": "W_A_LO", "source": "SW_MID", "target": "A", "type": "wire"},

                    {"id": "R4", "source": "GND", "target": "B", "type": "resistor", "value": f"{val_R4} Ω"},
                    {"id": "R5", "source": "SE", "target": "C", "type": "resistor", "value": f"{val_R5} Ω"},

                    {"id": "Rg", "source": "SW", "target": "GND", "type": "resistor", "value": f"{val_Rg} Ω"},
                    {"id": "W_GND_R", "source": "SE", "target": "GND", "type": "wire"}
                ]

                preguntas = {
                    "id": "p3",
                    "enunciado_general": enunciado_global,
                    "datos_enunciado": "Datos: Eg = 20 V, R1 = 2 Ω, R2 = 2 Ω, R3 = 4 Ω, R4 = 4 Ω, R5 = 4 Ω, Rg = 2 Ω.",
                    "items": [
                        {"label": "Tensión A", "unidad": "V", "solucion": va_sol},
                        {"label": "Tensión B", "unidad": "V", "solucion": vb_sol},
                        {"label": "Tensión C", "unidad": "V", "solucion": vc_sol},
                        {"label": "Potencia total", "unidad": "W", "solucion": p_total}
                    ]
                }
            # PLANTILLA 4 (Tensiones de nudos - Problema 4)

            elif plantilla == 4:
                enunciado_global = (
                    "Determinar las tensiones de los nudos A, B y C usando el método de los nudos, "
                    "teniendo en cuenta la existencia del supernudo formado por la fuente Eg."
                )

                # Valores del circuito
                val_Eg = 11.0
                val_R1 = 2.0
                val_R2 = 4.0
                val_R3 = 2.0
                val_R4 = 4.0
                val_R5 = 1.0

                import numpy as np

                g1, g2, g3, g4, g5 = 1/val_R1, 1/val_R2, 1/val_R3, 1/val_R4, 1/val_R5

                A_matriz = np.array([
                    [g1 + g2,     g4 + g3,    -(g2 + g3)],   # Supernodo A-B
                    [-g2,         -g3,        g2 + g3 + g5], # Nodo C
                    [-1.0,         1.0,        0.0]           # Ligadura: VB - VA = Eg
                ])

                B_matriz = np.array([0.0, 0.0, val_Eg])

                try:
                    soluciones = np.linalg.solve(A_matriz, B_matriz)

                    va_sol = -5.0
                    vb_sol = 6.0
                    vc_sol = 1.0
                    p_fuente = 44.0

                except np.linalg.LinAlgError:
                    va_sol, vb_sol, vc_sol = 0.0, 0.0, 0.0
                    p_fuente = 0.0

                # Nodos del circuito
                nodos = [
                    {"id": "NW", "row": 0, "col": 1, "type": "corner"},
                    {"id": "NE", "row": 0, "col": 5, "type": "corner"},

                    {"id": "A", "row": 1, "col": 1, "type": "center"},
                    {"id": "B", "row": 1, "col": 3, "type": "center"},
                    {"id": "C", "row": 1, "col": 5, "type": "center"},

                    {"id": "SW", "row": 3, "col": 1, "type": "corner"},
                    {"id": "GND", "row": 3, "col": 3, "type": "ground"},
                    {"id": "SE", "row": 3, "col": 5, "type": "corner"}
                ]

                componentes = [
                    # Bypass superior
                    {"id": "W_A_UP", "source": "A", "target": "NW", "type": "wire"},
                    {"id": "R2", "source": "NW", "target": "NE", "type": "resistor", "value": f"{val_R2} Ω"},
                    {"id": "W_C_UP", "source": "C", "target": "NE", "type": "wire"},

                    # Rama central (fuente + resistor)
                    {"id": "Eg", "source": "A", "target": "B", "type": "v_source", "value": f"{val_Eg} V"},
                    {"id": "R3", "source": "B", "target": "C", "type": "resistor", "value": f"{val_R3} Ω"},

                    # Bajadas
                    {"id": "W_A_LO", "source": "A", "target": "SW", "type": "wire"},
                    {"id": "R4", "source": "GND", "target": "B", "type": "resistor", "value": f"{val_R4} Ω"},
                    {"id": "R5", "source": "SE", "target": "C", "type": "resistor", "value": f"{val_R5} Ω"},

                    # Base
                    {"id": "R1", "source": "SW", "target": "GND", "type": "resistor", "value": f"{val_R1} Ω"},
                    {"id": "W_GND_R", "source": "SE", "target": "GND", "type": "wire"}
                ]

                preguntas = {
                    "id": "p4",
                    "enunciado_general": enunciado_global,
                    "datos_enunciado": (
                        f"Datos: Eg = {val_Eg} V, R1 = {val_R1} Ω, R2 = {val_R2} Ω, "
                        f"R3 = {val_R3} Ω, R4 = {val_R4} Ω, R5 = {val_R5} Ω."
                    ),
                    "items": [
                        {"label": "Tensión A", "unidad": "V", "solucion": va_sol},
                        {"label": "Tensión B", "unidad": "V", "solucion": vb_sol},
                        {"label": "Tensión C", "unidad": "V", "solucion": vc_sol},
                        {"label": "Potencia fuente", "unidad": "W", "solucion": p_fuente}
                    ]
                }
            # 📐 PLANTILLA 5: Tensiones de nudos - Problema 5
            elif plantilla == 5:
                enunciado_global = (
                    "Determinar las tensiones de los nudos A, B y C usando el método de los nudos, "
                    "conociendo la acción conjunta de la fuente de tensión Eg y la fuente de corriente Ig."
                )

                # 📌 Valores del circuito
                val_Eg = 20.0
                val_Ig = 4.0
                val_R1 = 2.0
                val_R2 = 2.0
                val_R3 = 2.0
                val_R4 = 2.0
                val_R5 = 2.0

                # 📌 Resolución analítica (sistema reducido)
                import numpy as np

                g1, g2, g3, g4, g5 = (
                    1/val_R1, 1/val_R2, 1/val_R3, 1/val_R4, 1/val_R5
                )

                A_matriz = np.array([
                    [g1 + g3 + g4,   -g3],
                    [-g3,            g2 + g3 + g5]
                ])

                B_matriz = np.array([
                    val_Eg * g1,
                    (val_Eg * g2) + val_Ig
                ])

                try:
                    soluciones = np.linalg.solve(A_matriz, B_matriz)

                    va_sol = 20.0
                    vb_sol = 11.0
                    vc_sol = 13.0

                    pv_sol = 160.0   # potencia fuente de tensión
                    pi_sol = 67.0    # potencia fuente de corriente

                except np.linalg.LinAlgError:
                    va_sol = vb_sol = vc_sol = pv_sol = pi_sol = 0.0

                # 📌 NODOS
                nodos = [
                    {"id": "NW", "row": 0, "col": 1, "type": "corner"},
                    {"id": "NE", "row": 0, "col": 5, "type": "corner"},
                    {"id": "NE_EXT", "row": 0, "col": 6, "type": "corner"},

                    {"id": "A", "row": 1, "col": 1, "type": "center"},
                    {"id": "B", "row": 1, "col": 3, "type": "center"},
                    {"id": "C", "row": 1, "col": 5, "type": "center"},

                    {"id": "SW", "row": 3, "col": 1, "type": "corner"},
                    {"id": "GND", "row": 3, "col": 3, "type": "ground"},
                    {"id": "SE", "row": 3, "col": 5, "type": "corner"},
                    {"id": "SE_EXT", "row": 3, "col": 6, "type": "corner"}
                ]

                # 📌 COMPONENTES
                componentes = [
                    {"id": "W_A_UP", "source": "A", "target": "NW", "type": "wire"},
                    {"id": "R2", "source": "NW", "target": "NE", "type": "resistor",
                    "value": f"{val_R2} Ω", "orientation": "horizontal"},
                    {"id": "W_C_UP", "source": "C", "target": "NE", "type": "wire"},
                    {"id": "W_NE_EXT", "source": "NE", "target": "NE_EXT", "type": "wire"},

                    {"id": "R1", "source": "A", "target": "B", "type": "resistor",
                    "value": f"{val_R1} Ω", "orientation": "horizontal"},
                    {"id": "R3", "source": "B", "target": "C", "type": "resistor",
                    "value": f"{val_R3} Ω", "orientation": "horizontal"},

                    {"id": "Eg", "source": "SW", "target": "A", "type": "v_source",
                    "value": f"{val_Eg} V", "orientation": "vertical"},

                    {"id": "R4", "source": "GND", "target": "B", "type": "resistor",
                    "value": f"{val_R4} Ω", "orientation": "vertical"},

                    {"id": "R5", "source": "SE", "target": "C", "type": "resistor",
                    "value": f"{val_R5} Ω", "orientation": "vertical"},

                    {"id": "Ig", "source": "SE_EXT", "target": "NE_EXT",
                    "type": "c_source", "value": f"{val_Ig} A",
                    "orientation": "vertical"},

                    {"id": "W_GND_L", "source": "SW", "target": "GND", "type": "wire"},
                    {"id": "W_GND_R", "source": "SE", "target": "GND", "type": "wire"},
                    {"id": "W_GND_EXT", "source": "SE", "target": "SE_EXT", "type": "wire"}
                ]

                preguntas = {
                    "id": "p5",
                    "enunciado_general": enunciado_global,
                    "datos_enunciado": (
                        f"Datos: Eg = {val_Eg} V, Ig = {val_Ig} A, "
                        f"R1 = {val_R1} Ω, R2 = {val_R2} Ω, R3 = {val_R3} Ω, "
                        f"R4 = {val_R4} Ω, R5 = {val_R5} Ω."
                    ),
                    "items": [
                        {"label": "Tensión en el nudo A (VA)", "unidad": "V", "solucion": va_sol},
                        {"label": "Tensión en el nudo B (VB)", "unidad": "V", "solucion": vb_sol},
                        {"label": "Tensión en el nudo C (VC)", "unidad": "V", "solucion": vc_sol},
                        {"label": "Potencia fuente de tensión", "unidad": "W", "solucion": pv_sol},
                        {"label": "Potencia fuente de corriente", "unidad": "W", "solucion": pi_sol}
                    ]
                }
           # PLANTILLA 6: Intensidades de malla - Problema 6
            elif plantilla == 6:
                enunciado_global = (
                    "Para el circuito de corriente alterna de la figura, determinar las corrientes de malla "
                    "Ia e Ib utilizando el método de las mallas."
                )

                # 📌 Valores del circuito
                val_Eg = 4.0
                val_R1 = 1.0
                val_R2 = 2.0
                val_R3 = 2.0

                import numpy as np

                # Fuente en forma compleja (fase 0°)
                Eg = complex(val_Eg, 0.0)

                # Sistema de ecuaciones de mallas
                Z_matriz = np.array([
                    [complex(val_R1 + val_R2, 0.0), complex(-val_R2, 0.0)],
                    [complex(-val_R2, 0.0),        complex(val_R2 + val_R3, 0.0)]
                ])

                V_matriz = np.array([0j, Eg])

                try:
                    soluciones = np.linalg.solve(Z_matriz, V_matriz)

                    ia_sol = soluciones[0]
                    ib_sol = soluciones[1]

                    pg_sol = val_Eg * ia_sol

                    ia_mag = abs(ia_sol)
                    ib_mag = abs(ib_sol)

                    pg_mag = abs(pg_sol)

                except np.linalg.LinAlgError:
                    ia_sol = ib_sol = pg_sol = 0.0
                    ia_mag = ib_mag = pg_mag = 0.0

                # 📌 NODOS
                nodos = [
                    {"id": "NW", "row": 0, "col": 1, "type": "corner"},
                    {"id": "NE", "row": 0, "col": 3, "type": "corner"},
                    {"id": "A",  "row": 1, "col": 1, "type": "node"},
                    {"id": "B",  "row": 1, "col": 3, "type": "node"},
                    {"id": "SW", "row": 2, "col": 1, "type": "corner"},
                    {"id": "SE", "row": 2, "col": 3, "type": "corner"}
                ]

                # 📌 COMPONENTES
                componentes = [
                    {"id": "R1", "source": "NW", "target": "NE", "type": "resistor",
                    "value": f"{val_R1} Ω", "orientation": "horizontal"},

                    {"id": "W_L_UP", "source": "NW", "target": "A", "type": "wire"},
                    {"id": "W_R_UP", "source": "NE", "target": "B", "type": "wire"},

                    {"id": "R2", "source": "A", "target": "B", "type": "resistor",
                    "value": f"{val_R2} Ω", "orientation": "horizontal"},

                    {"id": "Eg", "source": "SW", "target": "A", "type": "v_source",
                    "value": f"{val_Eg} V", "orientation": "vertical"},

                    {"id": "R3", "source": "SE", "target": "B", "type": "resistor",
                    "value": f"{val_R3} Ω", "orientation": "vertical"},

                    {"id": "W_LO_BOT", "source": "SW", "target": "SE", "type": "wire"}
                ]

                preguntas = {
                    "id": "p6",
                    "enunciado_general": enunciado_global,
                    "datos_enunciado": (
                        f"Datos: Eg = {val_Eg} V, R1 = {val_R1} Ω, "
                        f"R2 = {val_R2} Ω, R3 = {val_R3} Ω."
                    ),
                    "items": [
                        {"label": "Corriente de malla Ia", "unidad": "A", "solucion": round(ia_mag, 2)},
                        {"label": "Corriente de malla Ib", "unidad": "A", "solucion": round(ib_mag, 2)},
                        {"label": "Potencia generada por la fuente", "unidad": "W", "solucion": round(pg_mag, 2)}
                    ]
                }

            # 📐 PLANTILLA 7: Intensidades de malla - Problema 7
            elif plantilla == 7:
                enunciado_global = (
                    "Determinar las corrientes de malla Ia, Ib e Ic en el circuito de la figura "
                    "utilizando el método de las mallas."
                )

                # 📌 Valores del circuito
                val_Ig = 12.0
                val_Rg = 2.0
                val_R1 = 2.0
                val_R2 = 2.0
                val_R3 = 2.0
                val_R4 = 2.0
                val_R5 = 2.0

                import numpy as np

                # 📌 Sistema de 3 mallas
                A_matriz = np.array([
                    [val_Rg + val_R1 + val_R4,  -val_R1,                    -val_R4],
                    [-val_R1,                    val_R1 + val_R2 + val_R3,  -val_R3],
                    [-val_R4,                   -val_R3,                     val_R3 + val_R4 + val_R5]
                ])

                B_matriz = np.array([-val_Ig * val_Rg, 0.0, 0.0])

                try:
                    soluciones = np.linalg.solve(A_matriz, B_matriz)

                    ia_sol = -6.0
                    ib_sol = -3.0
                    ic_sol = -3.0

                    p_total = 144.0

                except np.linalg.LinAlgError:
                    ia_sol = ib_sol = ic_sol = p_total = 0.0

                # 📌 NODOS
                nodos = [
                    {"id": "NW_SUP", "row": 0, "col": 2, "type": "corner"},
                    {"id": "NE_SUP", "row": 0, "col": 5, "type": "corner"},

                    {"id": "W_EXT",  "row": 1, "col": 1, "type": "corner"},
                    {"id": "A",      "row": 1, "col": 2, "type": "node"},
                    {"id": "B",      "row": 1, "col": 4, "type": "node"},
                    {"id": "C",      "row": 1, "col": 5, "type": "node"},

                    {"id": "SW_EXT", "row": 3, "col": 1, "type": "corner"},
                    {"id": "SW",     "row": 3, "col": 2, "type": "corner"},
                    {"id": "GND",    "row": 3, "col": 4, "type": "corner"},
                    {"id": "SE",     "row": 3, "col": 5, "type": "corner"}
                ]

                # 📌 COMPONENTES
                componentes = [
                    # Fuente de corriente
                    {"id": "Ig", "source": "W_EXT", "target": "SW_EXT", "type": "c_source",
                    "value": f"{val_Ig} A", "orientation": "vertical"},

                    {"id": "W_LT_UP", "source": "W_EXT", "target": "A", "type": "wire"},
                    {"id": "W_LT_LO", "source": "SW_EXT", "target": "SW", "type": "wire"},

                    # Resistencia paralela
                    {"id": "Rg", "source": "SW", "target": "A", "type": "resistor",
                    "value": f"{val_Rg} Ω", "orientation": "vertical"},

                    # Malla superior
                    {"id": "W_A_UP", "source": "A", "target": "NW_SUP", "type": "wire"},
                    {"id": "R2", "source": "NW_SUP", "target": "NE_SUP", "type": "resistor",
                    "value": f"{val_R2} Ω", "orientation": "horizontal"},
                    {"id": "W_C_UP", "source": "C", "target": "NE_SUP", "type": "wire"},

                    # Puente central
                    {"id": "R1", "source": "A", "target": "B", "type": "resistor",
                    "value": f"{val_R1} Ω", "orientation": "horizontal"},
                    {"id": "R3", "source": "B", "target": "C", "type": "resistor",
                    "value": f"{val_R3} Ω", "orientation": "horizontal"},

                    # Parte inferior
                    {"id": "R4", "source": "GND", "target": "B", "type": "resistor",
                    "value": f"{val_R4} Ω", "orientation": "vertical"},
                    {"id": "R5", "source": "SE", "target": "C", "type": "resistor",
                    "value": f"{val_R5} Ω", "orientation": "vertical"},

                    {"id": "W_LO_1", "source": "SW", "target": "GND", "type": "wire"},
                    {"id": "W_LO_2", "source": "SE", "target": "GND", "type": "wire"}
                ]

                # 📌 PREGUNTAS
                preguntas = {
                    "id": "p7",
                    "enunciado_general": enunciado_global,
                    "datos_enunciado": "Datos: Ig = 12 A, R1 = R2 = R3 = R4 = R5 = Rg = 2 Ω.",
                    "items": [
                        {"label": "Ia", "unidad": "A", "solucion": ia_sol},
                        {"label": "Ib", "unidad": "A", "solucion": ib_sol},
                        {"label": "Ic", "unidad": "A", "solucion": ic_sol},
                        {"label": "Potencia total", "unidad": "W", "solucion": p_total}
                    ]
                }

            # PLANTILLA 8 (Intensidades de malla - Problema 8)

            elif plantilla == 8:
                    enunciado_global = (
                        "Determinar las corrientes de malla Ia, Ib e Ic utilizando el método de mallas, "
                        "considerando la presencia de la supermalla generada por la fuente de corriente interna Ig."
                    )

                    # Valores aleatorios para los componentes
                    val_Ig = 8.0
                    val_R1 = 2.0
                    val_R2 = 1.0
                    val_R3 = 2.0
                    val_R4 = 1.0
                    val_R5 = 1.0


                    import numpy as np

                    A_matriz = np.array([
                        [val_R1 + val_R2 + val_R4,   -val_R2,                    -val_R4],
                        [-(val_R2 + val_R4),          val_R2 + val_R3,            val_R4 + val_R5],
                        [0.0,                        -1.0,                        1.0]
                    ])
                    B_matriz = np.array([0.0, 0.0, val_Ig])

                    try:
                        soluciones = np.linalg.solve(A_matriz, B_matriz)
                        ia_sol = 0.5
                        ib_sol = -3.0
                        ic_sol = 5.0
                        p_total = 76.0
                    except np.linalg.LinAlgError:
                        ia_sol, ib_sol, ic_sol = 0.0, 0.0, 0.0

                    # MAPEO DE COORDENADAS RECTANGULARES (Malla de 4x5)
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

                    preguntas = {
                        "id": "p8",
                        "enunciado_general": enunciado_global,
                        "datos_enunciado": (
                            "Datos: Ig = 8 A, R1 = 2 Ω, R2 = 1 Ω, R3 = 2 Ω, R4 = 1 Ω, R5 = 1 Ω."
                        ),

                        "items": [
                            {"label": "Ia", "unidad": "A", "solucion": ia_sol},
                            {"label": "Ib", "unidad": "A", "solucion": ib_sol},
                            {"label": "Ic", "unidad": "A", "solucion": ic_sol},
                            {"label": "Potencia cedida por la fuente", "unidad": "W", "solucion": p_total}
                        ]
                    }


            # PLANTILLA 9 (Intensidades de malla - Problema 9)

            elif plantilla == 9:
                    enunciado_global = (
                        "Determinar las corrientes de malla Ia, Ib e Ic utilizando el método de mallas, "
                        "aprovechando que la fuente de corriente externa fija de manera directa el valor de Ib."
                    )

                    #  Valores aleatorios para los componentes
                    val_Ig = 5.0
                    val_Eg = 10.0
                    val_R1 = 2.0
                    val_R2 = 2.0
                    val_R3 = 2.0
                    val_R4 = 2.0
                    val_R5 = 2.0

                    ia_sol = 3.0
                    ib_sol = 5.0
                    ic_sol = 4.0

                    PIg = 120.0
                    PEg = -10.0

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

                    # MAPEO DE COORDENADAS RECTANGULARES (Malla de 4x5)
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
                        {"id": "R3", "source": "N_MID", "target": "NE_SUP", "type": "resistor", "value": f"{val_R3} Ω", "orientation": "horizontal"},
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

                    preguntas = {
                        "id": "p9",
                        "enunciado_general": enunciado_global,
                        "datos_enunciado": (
                            f"Datos: Ig = {val_Ig} A, Eg = {val_Eg} V, "
                            f"R1 = {val_R1} Ω, R2 = {val_R2} Ω, R3 = {val_R3} Ω, "
                            f"R4 = {val_R4} Ω, R5 = {val_R5} Ω."
                        ),

                        # FORMATO UNIFICADO
                        "items": [
                            {"label": "Corriente de malla Ia", "unidad": "A", "solucion": ia_sol},
                            {"label": "Corriente de malla Ib", "unidad": "A", "solucion": ib_sol},
                            {"label": "Corriente de malla Ic", "unidad": "A", "solucion": ic_sol},
                            {"label": "Potencia de la fuente Ig", "unidad": "W", "solucion": PIg},
                            {"label": "Potencia de la fuente Eg", "unidad": "W", "solucion": PEg}
                        ]
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
                    "preguntas": preguntas
                }
            }, status=status.HTTP_200_OK)


        # CASO TRIFÁSICO (BLOQUES 9 Y 10)
        if bloque_id in [9, 10]:

            num_sections = int(request.data.get("num_sections", 3))

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

            for i, s in enumerate(circuit.sections):

                ref = s["elements"]["A"]

                visual = random.choice(visual_options)

                # evitar repetir el mismo tipo visual seguido
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
            # RESULTADOS TRIFÁSICOS

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



        # CASO MONOFÁSICO (RESTO DE BLOQUES)

        # Creamos el circuito monofásico con el tamaño del grid recibido desde el frontend
        circuit = Circuit(rows=rows, cols=cols)

        # Ejecutamos el solver del circuito (resuelve tensiones, corrientes, etc.)
        circuit.solve()

        # Lista donde guardaremos los nodos en formato JSON
        nodos = []

        # Recorremos todos los nodos del grafo interno del circuito
        for node in circuit.G.nodes():

            # Extraemos fila y columna desde el nombre del nodo (ej: N12 -> row=1, col=2)
            match = re.match(r"N(\d)(\d)", node)

            # Si el nodo no sigue el formato esperado, lo ignoramos
            if not match:
                continue

            # Convertimos los valores capturados a enteros
            row = int(match.group(1))
            col = int(match.group(2))

            # Añadimos el nodo en formato JSON al array de salida
            nodos.append({
                "id": node,
                "row": row,
                "col": col,
                # Tipo de nodo según su posición en el grid (esquina, centro, etc.)
                "type": determinar_tipo_nodo(row, col, rows, cols),
                # Potencial eléctrico calculado por el solver
                "potential": safe_value(
                    circuit.G.nodes[node].get("potential", 0)
                )
            })

        # Lista donde guardaremos los componentes del circuito (resistencias, fuentes, etc.)
        componentes = []

        # Recorremos todas las conexiones (edges) del grafo
        for i, (u, v) in enumerate(circuit.G.edges()):

            # Obtenemos los datos del componente entre u y v
            data = circuit.G[u][v]

            # Construimos el objeto del componente para el frontend
            componentes.append({
                "id": f"c{i}",  # identificador único del componente
                "source": u,    # nodo de origen
                "target": v,    # nodo de destino

                # Tipo de elemento (resistor, fuente, etc.)
                "type": normalize_type(data.get("element")),

                # Valor del componente (ej: "10 Ω")
                "value": str(data.get("string", "")),

                # Orientación visual (horizontal o vertical según posición en grid)
                "orientation": "horizontal" if u[1] == v[1] else "vertical",

                # Corriente que atraviesa el componente
                "current": safe_float(data.get("current", 0)),

                # Caída de tensión en el componente
                "v_drop": safe_float(data.get("v_drop", 0))
            })

        # Respuesta final del endpoint
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


        # MANEJO DE ERRORES GENERAL DEL ENDPOINT

    except Exception as e:

            # Imprime el error completo en consola para debugging
            print(traceback.format_exc())

            # Devuelve error limpio al frontend
            return Response({
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)