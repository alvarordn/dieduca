"""
lib_new.py  –  Generador y resolvedor de circuitos de CA para autoevaluación.

Clases:
  Circuit           – Circuito monofásico CA (análisis nodal MNA).
  ThreePhaseCircuit – Circuito trifásico CA equilibrado (Y / Δ).

Funciones auxiliares:
  element_values()       – Lista de valores discretos entre dos límites.
  format_with_prefix()   – Formatea un valor con prefijo SI (m, µ, n…).
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import warnings


# ─────────────────────────────────────────────────────────────────────────────
#  Utilidades de formato
# ─────────────────────────────────────────────────────────────────────────────

# Prefijos SI ordenados de mayor a menor umbral de exponente positivo
_SI_PREFIXES = [
    (12, 'T'),
    (9,  'G'),
    (6,  'M'),
    (3,  'k'),
    (0,  ''),
    (-3, 'm'),
    (-6, 'µ'),
    (-9, 'n'),
    (-12,'p'),
]


def format_with_prefix(val: float, unit: str = '') -> str:
    """Devuelve *val* formateado con el prefijo SI más adecuado.

    Ejemplos:
        format_with_prefix(0.001, 'F')  →  '1 mF'
        format_with_prefix(1500, 'Ω')   →  '1.5 kΩ'
        format_with_prefix(1e-9, 'H')   →  '1 nH'
    """
    if val == 0:
        return f"0 {unit}".strip()

    exp = math.floor(math.log10(abs(val)))

    # Elegir el prefijo cuyo umbral sea ≤ exp (tomamos el mayor que cumple)
    chosen_exp, chosen_prefix = 0, ''
    for threshold, prefix in _SI_PREFIXES:
        if exp >= threshold:
            chosen_exp, chosen_prefix = threshold, prefix
            break

    scale = 10 ** chosen_exp
    scaled = val / scale
    return f"{scaled:g} {chosen_prefix}{unit}".strip()


def element_values(lb: float, ub: float) -> list:
    """Genera valores comerciales 1–2–5 entre *lb* y *ub* (inclusive)."""
    lb_exp = int(math.floor(math.log10(lb)))
    ub_exp = int(math.floor(math.log10(ub)))

    values = []
    for exp in range(lb_exp, ub_exp + 1):
        base = 10 ** exp
        for i in range(1, 10):
            val = i * base
            if lb <= val <= ub:
                values.append(val)
    return values


# ─────────────────────────────────────────────────────────────────────────────
#  Tablas de valores y tipos de elementos
# ─────────────────────────────────────────────────────────────────────────────

LIMITS = {
    'capacitor': element_values(1e-9, 1e-3),
    'inductor':  element_values(1e-4, 1e-3),
    'resistor':  element_values(1e-2, 1e+2),
    'v_source':  element_values(1e+2, 1e+4),
    'c_source':  element_values(1e-1, 1e+2),
}

PASSIVE_TYPES = ['capacitor', 'inductor', 'resistor', 'shortcircuit', 'opencircuit']
ACTIVE_TYPES  = ['v_source', 'c_source']

# Tipos cuya admitancia entra en la matriz nodal
ADMITTANCE_TYPES = {'capacitor', 'inductor', 'resistor'}
# Tipos que introducen una ecuación KVL extra (fuentes de tensión / cortocircuito)
VSOURCE_TYPES    = {'v_source', 'shortcircuit'}


# ─────────────────────────────────────────────────────────────────────────────
#  Clase Circuit  (monofásico CA)
# ─────────────────────────────────────────────────────────────────────────────

class Circuit:
    """Circuito de CA monofásico generado aleatoriamente sobre una cuadrícula.

    Parámetros
    ----------
    rows, cols : int
        Dimensiones de la cuadrícula de nodos.
    freq : float
        Frecuencia en Hz (por defecto 50 Hz).
    seed : int | None
        Semilla para reproducibilidad.
    max_retries : int
        Número máximo de intentos de regeneración si el circuito es singular.
    """

    def __init__(self, rows: int, cols: int, freq: float = 50,
                 seed: int | None = None, max_retries: int = 20):
        if rows < 2 or cols < 2:
            raise ValueError("Se necesitan al menos 2 filas y 2 columnas.")

        self.rows = rows
        self.cols = cols
        self.freq = freq
        self.omega = 2 * np.pi * freq
        self.solved = False

        if seed is not None:
            random.seed(seed)

        for attempt in range(max_retries):
            self.G = nx.DiGraph()
            self.nodes = [f"N{r}{c}" for r in range(rows) for c in range(cols)]
            self.G.add_nodes_from(self.nodes)
            self._add_edges(rows, cols)
            self._inject_sources()

            ok, reason = self._validate()
            if ok:
                break
        else:
            raise RuntimeError(
                f"No se pudo generar un circuito válido en {max_retries} intentos. "
                f"Último motivo: {reason}"
            )

    # ── Construcción ──────────────────────────────────────────────────────────

    def _impedance(self, el_type: str, val: float) -> complex | None:
        """Calcula la impedancia compleja de un elemento pasivo."""
        if el_type == 'resistor':
            return complex(val)
        if el_type == 'inductor':
            return complex(0, self.omega * val)
        if el_type == 'capacitor':
            return complex(0, -1.0 / (self.omega * val))
        return None  # shortcircuit / opencircuit / fuentes

    def _get_element_data(self, el_type: str) -> tuple:
        """Devuelve (valor_SI, string_legible, impedancia) para un elemento."""
        if el_type == 'shortcircuit':
            return None, 'SC', 0j
        if el_type == 'opencircuit':
            return None, 'OC', None

        val = random.choice(LIMITS[el_type])
        units = {'resistor': 'Ω', 'inductor': 'H', 'capacitor': 'F'}
        unit = units.get(el_type, '')
        string_val = format_with_prefix(val, unit)
        imp = self._impedance(el_type, val)
        return val, string_val, imp

    def _create_edge(self, u: str, v: str):
        el = random.choice(PASSIVE_TYPES)
        val, string_val, imp = self._get_element_data(el)
        self.G.add_edge(u, v, element=el, value=val, impedance=imp, string=string_val)

    def _add_edges(self, rows: int, cols: int):
        for r in range(rows):
            for c in range(cols):
                if c < cols - 1:
                    self._create_edge(f"N{r}{c}", f"N{r}{c+1}")
                if r < rows - 1:
                    self._create_edge(f"N{r}{c}", f"N{r+1}{c}")

    def _inject_sources(self):
        num_sources = max(int((2 * self.rows * self.cols - self.rows - self.cols) // 5), 1)
        edges_to_replace = random.sample(list(self.G.edges()), num_sources)
        for u, v in edges_to_replace:
            el = random.choice(ACTIVE_TYPES)
            val = random.choice(LIMITS[el])
            unit = 'V' if el == 'v_source' else 'A'
            self.G[u][v].update({
                'element':   el,
                'value':     val,
                'impedance': None,
                'string':    format_with_prefix(val, unit),
            })

    # ── Validación estructural ────────────────────────────────────────────────

    def _validate(self) -> tuple[bool, str]:
        """Comprueba condiciones necesarias de resolubilidad *antes* de resolver.

        Devuelve (True, '') si el circuito parece válido, o (False, motivo).
        """
        # 1. Conectividad: el grafo subyacente no dirigido debe ser conexo
        if not nx.is_connected(self.G.to_undirected()):
            return False, "Grafo no conexo"

        # 2. No puede haber SOLO fuentes de tensión / cortocircuitos en un corte
        #    (detección simple: ningún nodo puede tener únicamente vsources)
        for node in self.G.nodes():
            incident = (list(self.G.in_edges(node, data=True)) +
                        list(self.G.out_edges(node, data=True)))
            if incident and all(d['element'] in VSOURCE_TYPES for _, _, d in incident):
                return False, f"Nodo {node} rodeado solo de fuentes de tensión"

        # 3. Detección de bucle de fuentes de tensión (dos v_sources en paralelo puro)
        v_edges = [(u, v) for u, v, d in self.G.edges(data=True)
                   if d['element'] in VSOURCE_TYPES]
        v_pairs = set()
        for u, v in v_edges:
            key = tuple(sorted([u, v]))
            if key in v_pairs:
                return False, f"Dos fuentes de tensión en paralelo entre {key}"
            v_pairs.add(key)

        # 4. Verificación numérica rápida: ¿la matriz MNA tiene rango completo?
        A, _ = self._build_mna()
        if A.shape[0] == 0:
            return False, "Sistema vacío"
        rank = np.linalg.matrix_rank(A)
        if rank < A.shape[0]:
            return False, f"Matriz MNA singular (rango {rank} < {A.shape[0]})"

        return True, ''

    # ── Ensamblado MNA ────────────────────────────────────────────────────────

    def _build_mna(self) -> tuple[np.ndarray, np.ndarray]:
        """Ensambla la matriz MNA (Modified Nodal Analysis) y el vector RHS."""
        nodes = list(self.G.nodes())
        num_nodes = len(nodes)
        # Nodo 0 → tierra (potencial = 0), resto indexados 0…(N-2)
        node_map = {n: i - 1 for i, n in enumerate(nodes) if i > 0}

        v_sources = [(u, v, d) for u, v, d in self.G.edges(data=True)
                     if d['element'] in VSOURCE_TYPES]
        num_v = len(v_sources)
        size = (num_nodes - 1) + num_v

        A = np.zeros((size, size), dtype=complex)
        Z = np.zeros(size, dtype=complex)

        # Estampa de admitancias
        for u, v, data in self.G.edges(data=True):
            if data['element'] not in ADMITTANCE_TYPES:
                continue
            y = 1.0 / data['impedance']
            ui, vi = node_map.get(u), node_map.get(v)
            if ui is not None:
                A[ui, ui] += y
            if vi is not None:
                A[vi, vi] += y
            if ui is not None and vi is not None:
                A[ui, vi] -= y
                A[vi, ui] -= y

        # Fuentes de corriente
        for u, v, data in self.G.edges(data=True):
            if data['element'] != 'c_source':
                continue
            val = data['value']
            ui, vi = node_map.get(u), node_map.get(v)
            if ui is not None:
                Z[ui] -= val
            if vi is not None:
                Z[vi] += val

        # Fuentes de tensión y cortocircuitos (estampa KVL)
        for k, (u, v, data) in enumerate(v_sources):
            row = (num_nodes - 1) + k
            val = data['value'] if data['element'] == 'v_source' else 0j
            ui, vi = node_map.get(u), node_map.get(v)
            if ui is not None:
                A[row, ui] = -1
                A[ui, row] = -1
            if vi is not None:
                A[row, vi] = 1
                A[vi, row] = 1
            Z[row] = val

        return A, Z

    # ── Resolución ────────────────────────────────────────────────────────────

    def solve(self) -> bool:
        """Resuelve el circuito por MNA y anota tensiones, corrientes y potencias.

        Devuelve True si la resolución fue exitosa.
        """
        nodes = list(self.G.nodes())
        num_nodes = len(nodes)
        node_map = {n: i - 1 for i, n in enumerate(nodes) if i > 0}

        v_sources = [(u, v, d) for u, v, d in self.G.edges(data=True)
                     if d['element'] in VSOURCE_TYPES]

        A, Z = self._build_mna()

        try:
            X = np.linalg.solve(A, Z)
        except np.linalg.LinAlgError:
            warnings.warn("La matriz MNA es singular; circuito irresoluble.")
            return False

        # ── Potenciales nodales ──────────────────────────────────────────────
        vn = {n: (X[node_map[n]] if n in node_map else 0j) for n in nodes}
        for n in nodes:
            self.G.nodes[n]['potential'] = vn[n]

        # ── Tensiones, corrientes y potencias por rama ───────────────────────
        for u, v, data in self.G.edges(data=True):
            v_drop = vn[u] - vn[v]
            data['v_drop'] = v_drop
            elem = data['element']

            if elem in VSOURCE_TYPES:
                # La corriente es la variable auxiliar del sistema
                idx = next(k for k, (su, sv, sd) in enumerate(v_sources)
                           if su == u and sv == v and sd is data)
                I = X[(num_nodes - 1) + idx]

            elif elem == 'c_source':
                I = complex(data['value'])

            elif elem == 'opencircuit':
                I = 0j

            else:  # elementos pasivos con impedancia
                I = v_drop / data['impedance']

            data['current'] = I

            # Potencia compleja S = V · I*  (convenio receptor: P>0 consume)
            S = v_drop * np.conj(I)
            data['P'] = S.real   # potencia activa [W]
            data['Q'] = S.imag   # potencia reactiva [VAr]
            data['S'] = abs(S)   # potencia aparente [VA]

        self.solved = True
        return True

    # ── Visualización ─────────────────────────────────────────────────────────

    def draw(self, show_values: bool = True):
        """Dibuja el circuito con NetworkX/Matplotlib."""
        pos = nx.planar_layout(self.G)
        edge_labels = {
            (u, v): (data.get('string') or data['element'])
            for u, v, data in self.G.edges(data=True)
        } if show_values else nx.get_edge_attributes(self.G, 'element')

        plt.figure(figsize=(9, 7))
        nx.draw(
            self.G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=900,
            font_size=10,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20,
        )
        nx.draw_networkx_edge_labels(
            self.G, pos, edge_labels=edge_labels, font_color='darkred', font_size=8,
        )
        title = f"Circuito monofásico CA  –  f = {self.freq} Hz"
        if self.solved:
            title += "  [RESUELTO]"
        plt.title(title)
        # plt.tight_layout()
        plt.show()

    # ── Informe de resultados ─────────────────────────────────────────────────

    def report(self):
        """Imprime por pantalla tensiones nodales y resultados de cada rama."""
        if not self.solved:
            print("El circuito aún no ha sido resuelto. Llama a solve() primero.")
            return

        print("=" * 60)
        print("POTENCIALES NODALES")
        print("=" * 60)
        for n, data in self.G.nodes(data=True):
            V = data.get('potential', 0j)
            print(f"  {n}: {abs(V):.4f} V  ∠{np.degrees(np.angle(V)):.2f}°")

        print()
        print("=" * 60)
        print("RAMAS")
        print("=" * 60)
        for u, v, data in self.G.edges(data=True):
            elem = data['element']
            vd   = data.get('v_drop', 0j)
            I    = data.get('current', 0j)
            P    = data.get('P', 0.0)
            Q    = data.get('Q', 0.0)
            S    = data.get('S', 0.0)
            print(f"  {u}→{v}  [{elem:12s}  {data.get('string',''):>12}]"
                  f"  |V|={abs(vd):.3f} V"
                  f"  |I|={abs(I):.3f} A"
                  f"  P={P:.3f} W  Q={Q:.3f} VAr  S={S:.3f} VA")
        print()

    # ── Diagrama fasorial ─────────────────────────────────────────────────────

    def draw_phasors(self):
        """Diagrama fasorial de caídas de tensión por rama (plano complejo).

        Cada nodo queda situado en el plano fasorial según su potencial
        complejo V_n.  Para cada rama u→v se dibuja una flecha desde el
        punto V_v hasta el punto V_u; dicha flecha representa exactamente
        el fasor de caída de tensión V_drop = V_u − V_v, en magnitud y
        ángulo.  Las flechas de ramas que comparten nodos se encadenan
        automáticamente formando los polígonos cerrados de cada malla,
        verificando visualmente la KVL.

        El nodo de tierra (potencial = 0) aparece en el origen.
        """
        if not self.solved:
            print("El circuito aún no ha sido resuelto. Llama a solve() primero.")
            return

        # ── Paleta de colores por tipo de elemento ────────────────────────────
        ELEM_COLOR = {
            'resistor':    '#e05252',   # rojo
            'inductor':    '#5278e0',   # azul
            'capacitor':   '#52c07a',   # verde
            'v_source':    '#c07a00',   # naranja oscuro
            'c_source':    '#9b52e0',   # violeta
            'shortcircuit':'#333333',   # negro
            'opencircuit': '#aaaaaa',   # gris claro
        }

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axhline(0, color='lightgray', lw=0.8, zorder=0)
        ax.axvline(0, color='lightgray', lw=0.8, zorder=0)

        # Potenciales nodales en el plano complejo
        V_node = {n: d.get('potential', 0j) for n, d in self.G.nodes(data=True)}

        # ── Flechas de caída de tensión por rama ──────────────────────────────
        legend_entries = {}
        for u, v, data in self.G.edges(data=True):
            Vu = V_node[u]
            Vv = V_node[v]
            # El fasor V_drop va desde Vv (cola) hasta Vu (punta)
            tail = (Vv.real, Vv.imag)
            head = (Vu.real, Vu.imag)
            dx   = head[0] - tail[0]
            dy   = head[1] - tail[1]

            elem  = data['element']
            color = ELEM_COLOR.get(elem, '#555555')
            vd    = data.get('v_drop', 0j)
            label_elem = elem if elem not in legend_entries else None

            # Flecha principal
            ax.annotate(
                '', xy=head, xytext=tail,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8),
                zorder=2,
            )
            # Etiqueta en el punto medio de la flecha
            if abs(vd) > 1e-9:
                mx = (tail[0] + head[0]) / 2
                my = (tail[1] + head[1]) / 2
                lbl = (f"{u}→{v}\n"
                       f"{data.get('string', elem)}\n"
                       f"|V|={abs(vd):.2f}V ∠{np.degrees(np.angle(vd)):.1f}°")
                ax.text(mx, my, lbl, fontsize=6.5, color=color,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                  ec=color, alpha=0.75, lw=0.6),
                        zorder=3)

            if label_elem:
                legend_entries[elem] = plt.Line2D(
                    [], [], color=color, lw=2, label=elem)

        # ── Puntos y etiquetas de nodos ───────────────────────────────────────
        for node, V in V_node.items():
            ax.plot(V.real, V.imag, 'o', color='black', ms=5, zorder=4)
            offset_x = 0.0
            offset_y = max(abs(vv) for vv in V_node.values()) * 0.03 + 1e-6
            ax.text(V.real + offset_x, V.imag + offset_y,
                    f"{node}\n{abs(V):.1f}V ∠{np.degrees(np.angle(V)):.1f}°",
                    fontsize=7, ha='center', color='#222222', zorder=5)

        # # ── Leyenda, ejes y título ────────────────────────────────────────────
        if legend_entries:
            ax.legend(handles=list(legend_entries.values()),
                      fontsize=8, loc='best', framealpha=0.8)

        ax.set_xlabel("Re  [V]", fontsize=9)
        ax.set_ylabel("Im  [V]", fontsize=9)
        ax.set_title(
            f"Diagrama fasorial de tensiones – monofásico CA  f={self.freq} Hz\n"
            f"Cada flecha u→v representa la caída V_u − V_v (KVL por malla)",
            fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Clase ThreePhaseCircuit  (CA trifásico equilibrado)
# ─────────────────────────────────────────────────────────────────────────────
# Colores por fase para dibujo
_PHASE_COLOR = {'A': '#c0392b', 'B': '#2471a3', 'C': '#1e8449'}
_PHASES      = ['A', 'B', 'C']

# Ángulos de secuencia positiva A-B-C
_PHASE_ANGLES = {'A': 0.0, 'B': -2*np.pi/3, 'C': 2*np.pi/3}


class ThreePhaseCircuit:
    """Circuito trifásico CA equilibrado en una sola fila de secciones en serie.

    La red se construye como una cadena de *num_sections* secciones, donde
    cada sección es, de forma aleatoria:

    * **serie**    – un único elemento trifásico (R, L o C) en serie con
                     cada línea de fase.
    * **paralelo** – un grupo de tres impedancias en paralelo con las líneas,
                     conectadas aleatoriamente en **estrella (Y)** o
                     **triángulo (Δ)**.

    La fuente de alimentación está siempre a la izquierda (columna 0) y la
    carga total equivalente se calcula como la combinación serie/paralelo de
    todas las secciones.

    Parámetros
    ----------
    num_sections : int
        Número de secciones de la cadena (2…6 recomendado).
    freq : float
        Frecuencia en Hz.
    v_line : float
        Tensión de línea eficaz [V].
    seed : int | None
        Semilla aleatoria.
    """

    def __init__(self, num_sections: int = 3, freq: float = 50,
                 v_line: float = 400, seed: int | None = None):
        if num_sections < 1:
            raise ValueError("num_sections debe ser ≥ 1.")

        self.num_sections = num_sections
        self.freq         = freq
        self.omega        = 2 * np.pi * freq
        self.v_line       = v_line
        self.v_phase      = v_line / np.sqrt(3)
        self.solved       = False

        if seed is not None:
            random.seed(seed)

        self.sections = [self._random_section(i) for i in range(num_sections)]

    # ── Generación de elementos ───────────────────────────────────────────────

    def _random_element(self) -> dict:
        """Devuelve un dict con los datos de un elemento pasivo aleatorio."""
        el  = random.choice(['resistor', 'inductor', 'capacitor'])
        val = random.choice(LIMITS[el])
        units = {'resistor': 'Ω', 'inductor': 'H', 'capacitor': 'F'}
        string = format_with_prefix(val, units[el])
        if el == 'resistor':
            Z = complex(val)
        elif el == 'inductor':
            Z = complex(0, self.omega * val)
        else:
            Z = complex(0, -1.0 / (self.omega * val))
        return {'element': el, 'value': val, 'impedance': Z, 'string': string}

    def _random_section(self, idx: int) -> dict:
        """Genera una sección aleatoria (serie o paralelo-Y / paralelo-Δ)."""
        kind = random.choice(['serie', 'paralelo'])
        if kind == 'serie':
            # Un elemento por fase (misma impedancia en las tres → equilibrado)
            elem = self._random_element()
            return {
                'kind':      'serie',
                'idx':       idx,
                'elements':  {ph: elem for ph in _PHASES},   # mismo Z por simetría
                'Z_phase':   elem['impedance'],               # Z vista por fase
            }
        else:
            # Paralelo en Y o Δ, elegido al azar
            conn = random.choice(['Y', 'delta'])
            elem = self._random_element()
            if conn == 'Y':
                # Z_fase_equivalente = Z_Y  (cada brazo conectado entre línea y neutro)
                Z_eq = elem['impedance']
            else:
                # Δ → Y equivalente: Z_Y = Z_Δ / 3
                Z_eq = elem['impedance'] / 3
            return {
                'kind':      'paralelo',
                'conn':      conn,
                'idx':       idx,
                'elements':  {ph: elem for ph in _PHASES},
                'Z_phase':   Z_eq,
            }

    # ── Impedancia total equivalente ──────────────────────────────────────────

    def _total_Z_phase(self) -> complex:
        """Impedancia de fase total: suma serie de todas las secciones."""
        return sum(s['Z_phase'] for s in self.sections)

    # ── Fasores de referencia ─────────────────────────────────────────────────

    def _V_phase(self, ph: str) -> complex:
        return self.v_phase * np.exp(1j * _PHASE_ANGLES[ph])

    def _V_line(self, ph_from: str, ph_to: str) -> complex:
        return self._V_phase(ph_from) - self._V_phase(ph_to)

    # ── Resolución ────────────────────────────────────────────────────────────

    def solve(self) -> bool:
        """Resuelve el circuito por fase y calcula P, Q, S en cada sección."""
        Z_total = self._total_Z_phase()
        if abs(Z_total) < 1e-12:
            warnings.warn("Impedancia total nula.")
            return False

        self.results = {}

        for ph in _PHASES:
            V_f  = self._V_phase(ph)
            I_L  = V_f / Z_total          # corriente de línea (circuito serie equiv.)

            # Tensión y potencia por sección
            secs = []
            for s in self.sections:
                V_s = I_L * s['Z_phase']  # caída en la sección (vista desde una fase)

                if s['kind'] == 'serie':
                    I_elem = I_L
                    V_elem = V_s
                else:  # paralelo
                    # La corriente que circula por cada brazo del paralelo
                    if s['conn'] == 'Y':
                        I_elem = I_L           # toda la corriente de línea
                        V_elem = V_s           # caída = V en cada brazo Y
                    else:  # Δ
                        # Tensión de línea entre las dos fases que forma el brazo Δ
                        # (para fase A el brazo es AB → V_AB)
                        pairs = {'A': ('A','B'), 'B': ('B','C'), 'C': ('C','A')}
                        p1, p2 = pairs[ph]
                        V_elem = self._V_line(p1, p2)
                        I_elem = V_elem / s['elements'][ph]['impedance']

                S_s = V_elem * np.conj(I_elem)
                secs.append({
                    'section':  s,
                    'V':        V_elem,
                    'I':        I_elem,
                    'P':        S_s.real,
                    'Q':        S_s.imag,
                    'S':        abs(S_s),
                })

            S_total_ph = sum(ss['P'] for ss in secs) + 1j*sum(ss['Q'] for ss in secs)
            self.results[ph] = {
                'V_phase':  V_f,
                'I_line':   I_L,
                'sections': secs,
                'P':        S_total_ph.real,
                'Q':        S_total_ph.imag,
                'S':        abs(S_total_ph),
            }

        self.P_total = sum(self.results[ph]['P'] for ph in _PHASES)
        self.Q_total = sum(self.results[ph]['Q'] for ph in _PHASES)
        self.S_total = sum(self.results[ph]['S'] for ph in _PHASES)
        self.pf      = self.P_total / self.S_total if self.S_total else 0

        self.solved = True
        return True

    # ── Informe ───────────────────────────────────────────────────────────────

    def report(self):
        if not self.solved:
            print("Llama a solve() primero.")
            return

        print("=" * 70)
        print(f"CIRCUITO TRIFÁSICO CA  –  f={self.freq} Hz  "
              f"V_línea={self.v_line} V  V_fase={self.v_phase:.2f} V")
        print(f"  Z_total_fase = {self._total_Z_phase():.4f} Ω")
        print("=" * 70)

        for ph in _PHASES:
            r = self.results[ph]
            print(f"\n  Fase {ph}:  |V|={abs(r['V_phase']):.2f} V ∠{np.degrees(np.angle(r['V_phase'])):.1f}°"
                  f"   |I_L|={abs(r['I_line']):.4f} A ∠{np.degrees(np.angle(r['I_line'])):.1f}°")
            for i, ss in enumerate(r['sections']):
                s    = ss['section']
                kind = s['kind']
                conn = f" ({s['conn']})" if kind == 'paralelo' else ''
                elem = s['elements'][ph]
                print(f"    Sec.{i}  [{kind}{conn:6s}]  {elem['string']:>10}"
                      f"  |V|={abs(ss['V']):.3f} V"
                      f"  |I|={abs(ss['I']):.4f} A"
                      f"  P={ss['P']:.2f} W  Q={ss['Q']:.2f} VAr")

        print()
        print(f"  P_total = {self.P_total:.2f} W")
        print(f"  Q_total = {self.Q_total:.2f} VAr")
        print(f"  S_total = {self.S_total:.2f} VA")
        print(f"  FP      = {self.pf:.4f}")
        print()

    # ── Dibujo del esquema ────────────────────────────────────────────────────

    def draw(self):
        """Dibuja el esquema unifilar trifásico en una cuadrícula rectangular.

        Convención de layout:
        * Columnas = nodos de la cadena (0 … num_sections).
        * Filas    = fases A (arriba), B (centro), C (abajo).
        * Las tres líneas horizontales van de izquierda a derecha.
        * Sección serie   → un rectángulo en cada línea de fase.
        * Sección paralelo-Y → un triángulo invertido con tres ramas
                               que convergen en el neutro (punto central).
        * Sección paralelo-Δ → un triángulo entre las tres líneas.
        """
        NS  = self.num_sections
        # Coordenadas y de las tres fases
        Y   = {'A': 2.0, 'B': 1.0, 'C': 0.0}
        # Coordenadas x de los nodos de columna
        X   = [i * 2.5 for i in range(NS + 1)]
        # x central de cada sección
        XM  = [(X[i] + X[i+1]) / 2 for i in range(NS)]

        fig, ax = plt.subplots(figsize=(2.5 + NS * 2.5, 4))
        ax.set_xlim(-0.8, X[-1] + 0.8)
        ax.set_ylim(-0.8, 3.0)
        ax.set_aspect('equal')
        ax.axis('off')

        ELEM_COLOR = {
            'resistor':  '#c0392b',
            'inductor':  '#2471a3',
            'capacitor': '#1e8449',
        }
        BOX_W, BOX_H = 0.7, 0.28   # dimensiones del rectángulo de elemento

        def draw_element(cx, cy, elem_dict, phase):
            """Dibuja un rectángulo con el símbolo del elemento."""
            color = ELEM_COLOR.get(elem_dict['element'], '#555')
            rect = plt.Rectangle((cx - BOX_W/2, cy - BOX_H/2),
                                  BOX_W, BOX_H,
                                  linewidth=1.5, edgecolor=color,
                                  facecolor='white', zorder=3)
            ax.add_patch(rect)
            ax.text(cx, cy, elem_dict['string'],
                    ha='center', va='center', fontsize=6.5,
                    color=color, zorder=4)

        # ── Etiquetas de fase (extremo izquierdo) ────────────────────────────
        for ph in _PHASES:
            ax.text(X[0] - 0.5, Y[ph], ph,
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color=_PHASE_COLOR[ph])

        # ── Secciones ─────────────────────────────────────────────────────────
        GAP = 0.55   # semiancho de la zona reservada para el paralelo

        for i, s in enumerate(self.sections):
            xm     = XM[i]
            x_left = X[i]
            x_rght = X[i+1]
            elem_A = s['elements']['A']   # mismo elemento en las tres fases

            if s['kind'] == 'serie':
                # Línea horizontal completa + rectángulo centrado en cada fase
                for ph in _PHASES:
                    ax.plot([x_left, x_rght], [Y[ph], Y[ph]],
                            color=_PHASE_COLOR[ph], lw=1.5, zorder=1)
                    draw_element(xm, Y[ph], elem_A, ph)

            elif s['conn'] == 'Y':
                # ── Estrella: tres ramas verticales hacia el neutro central ──
                xn = xm
                yn = Y['B']   # neutro en la altura de la fase B (centro)

                for ph in _PHASES:
                    y_ph   = Y[ph]
                    # punto de derivación sobre la línea horizontal
                    x_tap  = xn
                    # interrumpir la línea horizontal en la zona del paralelo
                    ax.plot([x_left, x_tap], [y_ph, y_ph],
                            color=_PHASE_COLOR[ph], lw=1.5, zorder=1)
                    ax.plot([x_tap, x_rght], [y_ph, y_ph],
                            color=_PHASE_COLOR[ph], lw=1.5, zorder=1)
                    # punto de tap
                    ax.plot(x_tap, y_ph, 'o', color=_PHASE_COLOR[ph], ms=4, zorder=4)

                    # rama vertical: línea de fase → elemento → neutro
                    y_mid = (y_ph + yn) / 2
                    ax.plot([x_tap, x_tap], [y_ph, y_mid + BOX_H/2],
                            color='gray', lw=1.0, zorder=2)
                    ax.plot([x_tap, x_tap], [y_mid - BOX_H/2, yn],
                            color='gray', lw=1.0, zorder=2)
                    draw_element(x_tap, y_mid, elem_A, ph)

                # Punto de neutro
                ax.plot(xn, yn, 'o', color='black', ms=5, zorder=5)
                ax.text(xn + 0.15, yn - 0.18, 'N', fontsize=7, va='center')
                ax.text(xm, 2.55, f'Sec.{i}  ∥-Y',
                        ha='center', fontsize=7, color='#555')

            else:  # Δ
                # ── Triángulo: vértices sobre las propias líneas de fase ──────
                # Cada vértice se desplaza horizontalmente para separar los lados
                # y que el triángulo no degenere en un segmento vertical.
                #   Fase A (y=2): vértice en xm (centro)
                #   Fase B (y=1): vértice en xm - GAP (izquierda)
                #   Fase C (y=0): vértice en xm + GAP (derecha)
                # Así los tres lados A-B, B-C, C-A forman un triángulo real.
                verts = {
                    'A': (xm,        Y['A']),
                    'B': (xm - GAP,  Y['B']),
                    'C': (xm + GAP,  Y['C']),
                }

                for ph in _PHASES:
                    y_ph      = Y[ph]
                    xv, _     = verts[ph]
                    # Tramo horizontal izquierdo: desde borde sección hasta el vértice
                    ax.plot([x_left, xv], [y_ph, y_ph],
                            color=_PHASE_COLOR[ph], lw=1.5, zorder=1)
                    # Tramo horizontal derecho: desde el vértice hasta borde sección
                    ax.plot([xv, x_rght], [y_ph, y_ph],
                            color=_PHASE_COLOR[ph], lw=1.5, zorder=1)
                    # Punto de vértice sobre la línea
                    ax.plot(xv, y_ph, 'o', color=_PHASE_COLOR[ph], ms=5, zorder=5)

                # Lados del triángulo con elemento en el centro de cada lado
                sides = [('A', 'B'), ('B', 'C'), ('C', 'A')]
                for ph1, ph2 in sides:
                    x1v, y1v = verts[ph1]
                    x2v, y2v = verts[ph2]
                    xc = (x1v + x2v) / 2
                    yc = (y1v + y2v) / 2
                    ax.plot([x1v, x2v], [y1v, y2v],
                            color='gray', lw=1.2, zorder=2)
                    draw_element(xc, yc, elem_A, ph1)

                ax.text(xm, 2.55, f'Sec.{i}  ∥-Δ',
                        ha='center', fontsize=7, color='#555')

        # ── Fuente (barras en el extremo izquierdo) ───────────────────────────
        for ph in _PHASES:
            ax.plot([X[0], X[0]], [Y[ph] - 0.15, Y[ph] + 0.15],
                    color=_PHASE_COLOR[ph], lw=3, zorder=3)
        ax.text(X[0], -0.55,
                f'{self.v_line} V\n{self.freq} Hz',
                ha='center', fontsize=8, color='#333')

        ax.set_title(
            f"Circuito trifásico CA equilibrado  –  "
            f"f={self.freq} Hz  V_L={self.v_line} V",
            fontsize=10, pad=8)
        plt.tight_layout()
        plt.show()

    # ── Diagrama fasorial ─────────────────────────────────────────────────────

    def draw_phasors(self):
        """Diagrama fasorial de tensiones de fase y caídas por sección.

        En cada fase se encadenan los fasores de caída de tensión de cada
        sección (tip-to-tail) desde el origen hasta el fasor de tensión de
        fase completo, verificando visualmente la KVL.
        """
        if not self.solved:
            print("Llama a solve() primero.")
            return

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal')
        ax.axhline(0, color='lightgray', lw=0.8, zorder=0)
        ax.axvline(0, color='lightgray', lw=0.8, zorder=0)

        ELEM_COLOR = {'resistor': '#c0392b', 'inductor': '#2471a3',
                      'capacitor': '#1e8449'}

        for ph in _PHASES:
            ph_color = _PHASE_COLOR[ph]
            r = self.results[ph]

            # Fasor de tensión de fase total (desde el origen)
            V_f = r['V_phase']
            ax.annotate('', xy=(V_f.real, V_f.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=ph_color,
                                        lw=2.5, linestyle='dashed'),
                        zorder=2)
            ax.text(V_f.real * 1.04, V_f.imag * 1.04,
                    f"V_{ph}  {abs(V_f):.1f}V ∠{np.degrees(np.angle(V_f)):.1f}°",
                    color=ph_color, fontsize=8, fontweight='bold', zorder=5)

            # Fasores de caída por sección encadenados
            cursor = complex(0, 0)
            for ss in r['sections']:
                s    = ss['section']
                V_s  = ss['V']
                tail = (cursor.real, cursor.imag)
                tip  = (cursor.real + V_s.real, cursor.imag + V_s.imag)
                elem = s['elements'][ph]
                color = ELEM_COLOR.get(elem['element'], '#888')

                ax.annotate('', xy=tip, xytext=tail,
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.8),
                            zorder=3)
                mx = (tail[0] + tip[0]) / 2
                my = (tail[1] + tip[1]) / 2
                kind_lbl = s['kind'] if s['kind'] == 'serie' else f"∥-{s['conn']}"
                lbl = f"S{s['idx']} {kind_lbl}\n{elem['string']}\n{abs(V_s):.2f}V"
                ax.text(mx, my, lbl, fontsize=6, color=color, ha='center',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                  ec=color, alpha=0.75, lw=0.5), zorder=4)
                cursor += V_s

        ax.set_xlabel("Re  [V]", fontsize=9)
        ax.set_ylabel("Im  [V]", fontsize=9)
        ax.set_title(
            f"Diagrama fasorial – trifásico CA  f={self.freq} Hz\n"
            f"Fasores de caída encadenados por sección (KVL por fase)",
            fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.show()