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
        plt.tight_layout()
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

    def draw_phasors(self, show_currents: bool = True, max_branches: int = 12):
        """Diagrama fasorial de tensiones nodales y corrientes de rama.

        Parámetros
        ----------
        show_currents : bool
            Si True, superpone los fasores de corriente de cada rama
            (escalados para ser comparables con las tensiones).
        max_branches : int
            Número máximo de ramas de corriente a dibujar (evita
            saturar el gráfico en circuitos grandes).

        Notas
        -----
        * El eje de referencia (ángulo 0) corresponde al fasor de la
          fuente de tensión principal, o simplemente al eje real si no
          hay ninguna.
        * Las tensiones nodales se dibujan desde el origen.
        * Las corrientes de rama se escalan al 80 % del módulo máximo
          de tensión para que sean visibles en la misma figura.
        * Los fasores nulos (módulo < 1e-12) se omiten.
        """
        if not self.solved:
            print("El circuito aún no ha sido resuelto. Llama a solve() primero.")
            return

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal')
        ax.axhline(0, color='lightgray', lw=0.8)
        ax.axvline(0, color='lightgray', lw=0.8)

        arrow_kw = dict(head_width=0, head_length=0, length_includes_head=True)

        # ── Fasores de tensión nodal ──────────────────────────────────────────
        node_colors = plt.cm.tab10.colors
        v_max = max(
            (abs(d.get('potential', 0j)) for _, d in self.G.nodes(data=True)),
            default=1.0
        ) or 1.0

        plotted_nodes = []
        for k, (node, data) in enumerate(self.G.nodes(data=True)):
            V = data.get('potential', 0j)
            if abs(V) < 1e-12:
                continue  # tierra u nodo a potencial nulo
            color = node_colors[k % len(node_colors)]
            ax.annotate(
                '', xy=(V.real, V.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0),
            )
            ax.text(V.real * 1.05, V.imag * 1.05,
                    f"{node}\n|V|={abs(V):.1f}V ∠{np.degrees(np.angle(V)):.1f}°",
                    color=color, fontsize=7.5, ha='center')
            plotted_nodes.append(node)

        # ── Fasores de corriente de rama (escalados) ──────────────────────────
        if show_currents:
            branches = [(u, v, d) for u, v, d in self.G.edges(data=True)
                        if abs(d.get('current', 0j)) > 1e-12]
            # Limitar número de ramas para no saturar el gráfico
            if len(branches) > max_branches:
                branches = branches[:max_branches]

            i_max = max((abs(d['current']) for _, _, d in branches), default=1.0) or 1.0
            scale = 0.8 * v_max / i_max   # escala A → V para visualización

            for u, v, data in branches:
                I = data['current'] * scale
                elem  = data['element']
                label = f"{u}→{v}\n|I|={abs(data['current']):.3f}A ∠{np.degrees(np.angle(data['current'])):.1f}°"
                ax.annotate(
                    '', xy=(I.real, I.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.0,
                                    linestyle='dashed'),
                )
                ax.text(I.real * 1.05, I.imag * 1.05, label,
                        color='gray', fontsize=6.5, ha='center')

        ax.set_xlabel("Re  [V]")
        ax.set_ylabel("Im  [V]")
        title = (f"Diagrama fasorial – monofásico CA  f={self.freq} Hz\n"
                 f"Tensiones nodales" +
                 ("  +  Corrientes de rama (escaladas)" if show_currents else ""))
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Clase ThreePhaseCircuit  (CA trifásico equilibrado)
# ─────────────────────────────────────────────────────────────────────────────

class ThreePhaseCircuit:
    """Circuito de CA trifásico equilibrado (secuencia positiva, A-B-C).

    Genera automáticamente una carga trifásica con impedancias de fase
    aleatorias en configuración Y (estrella) o Δ (triángulo) y calcula
    tensiones, corrientes y potencias por fase y totales.

    Parámetros
    ----------
    freq : float
        Frecuencia en Hz (por defecto 50 Hz).
    v_line : float
        Tensión de línea eficaz en V (por defecto 400 V).
    connection : {'Y', 'delta'}
        Configuración de la carga.
    seed : int | None
        Semilla para reproducibilidad.
    """

    # Fasores de secuencia positiva A-B-C (en radianes)
    _PHASE_ANGLES = {'A': 0, 'B': -2 * np.pi / 3, 'C': 2 * np.pi / 3}

    def __init__(self, freq: float = 50, v_line: float = 400,
                 connection: str = 'Y', seed: int | None = None):
        if connection not in ('Y', 'delta'):
            raise ValueError("connection debe ser 'Y' o 'delta'.")

        self.freq       = freq
        self.omega      = 2 * np.pi * freq
        self.v_line     = v_line          # tensión de línea (V_LL)
        self.v_phase    = v_line / np.sqrt(3)  # tensión de fase (V_LN)
        self.connection = connection
        self.solved     = False

        if seed is not None:
            random.seed(seed)

        self._generate_load()

    # ── Construcción ──────────────────────────────────────────────────────────

    def _random_passive_impedance(self) -> tuple[str, float, complex, str]:
        """Genera un elemento pasivo aleatorio y devuelve (tipo, val, Z, string)."""
        el = random.choice(['resistor', 'inductor', 'capacitor'])
        val = random.choice(LIMITS[el])
        units = {'resistor': 'Ω', 'inductor': 'H', 'capacitor': 'F'}
        string = format_with_prefix(val, units[el])

        if el == 'resistor':
            Z = complex(val)
        elif el == 'inductor':
            Z = complex(0, self.omega * val)
        else:
            Z = complex(0, -1.0 / (self.omega * val))
        return el, val, Z, string

    def _generate_load(self):
        """Genera impedancias de carga para las tres fases (equilibrado: Z_A=Z_B=Z_C)."""
        el, val, Z, string = self._random_passive_impedance()
        # Circuito equilibrado: misma impedancia en las tres fases
        self.load = {'element': el, 'value': val, 'impedance': Z, 'string': string}

    # ── Fasores de tensión ────────────────────────────────────────────────────

    def _v_phase_phasor(self, phase: str) -> complex:
        """Fasor de tensión de fase (V_LN) para la fase indicada."""
        angle = self._PHASE_ANGLES[phase]
        return self.v_phase * np.exp(1j * angle)

    def _v_line_phasor(self, from_ph: str, to_ph: str) -> complex:
        """Fasor de tensión de línea entre dos fases."""
        return self._v_phase_phasor(from_ph) - self._v_phase_phasor(to_ph)

    # ── Resolución ────────────────────────────────────────────────────────────

    def solve(self) -> bool:
        """Calcula tensiones, corrientes y potencias del sistema trifásico.

        En conexión Y la tensión de fase es V_LN y la corriente de fase
        es I = V_LN / Z.  En conexión Δ la tensión sobre la carga es V_LL
        y la corriente de fase es I_fase = V_LL / Z; la corriente de línea
        es I_L = √3 · I_fase.
        """
        Z = self.load['impedance']
        if Z == 0:
            warnings.warn("Impedancia de carga nula (cortocircuito).")
            return False

        phases = ['A', 'B', 'C']
        self.results = {}

        if self.connection == 'Y':
            for ph in phases:
                V_fase = self._v_phase_phasor(ph)      # tensión nodo-neutro
                I_linea = V_fase / Z                   # = I de fase en Y
                S = V_fase * np.conj(I_linea)
                self.results[ph] = {
                    'V_phase': V_fase,
                    'I_phase': I_linea,
                    'I_line':  I_linea,
                    'P': S.real,
                    'Q': S.imag,
                    'S': abs(S),
                }

        else:  # Δ
            line_pairs = [('A', 'B'), ('B', 'C'), ('C', 'A')]
            phase_currents = {}
            for ph_from, ph_to in line_pairs:
                V_ll = self._v_line_phasor(ph_from, ph_to)
                I_fase = V_ll / Z
                S = V_ll * np.conj(I_fase)
                key = f"{ph_from}{ph_to}"
                phase_currents[key] = {
                    'V_delta': V_ll,
                    'I_phase': I_fase,
                    'P': S.real,
                    'Q': S.imag,
                    'S': abs(S),
                }

            # Corrientes de línea: I_A = I_AB - I_CA, etc.
            for ph in phases:
                pairs = line_pairs          # (A→B, B→C, C→A)
                # Corriente de línea de fase ph = suma algebraica de corrientes Δ incidentes
                out_pair = next(f"{a}{b}" for a, b in pairs if a == ph)
                in_pair  = next(f"{a}{b}" for a, b in pairs if b == ph)
                I_L = phase_currents[out_pair]['I_phase'] - phase_currents[in_pair]['I_phase']
                self.results[ph] = {
                    'I_line': I_L,
                }
            self.results['delta'] = phase_currents

        # Potencias totales (equilibrado → × 3)
        sample_ph = self.results.get('A') or next(iter(self.results.values()))
        self.P_total = 3 * sample_ph['P']
        self.Q_total = 3 * sample_ph['Q']
        self.S_total = 3 * sample_ph['S']
        self.pf      = self.P_total / self.S_total if self.S_total else 0

        self.solved = True
        return True

    # ── Informe ───────────────────────────────────────────────────────────────

    def report(self):
        """Imprime los resultados del sistema trifásico."""
        if not self.solved:
            print("El circuito aún no ha sido resuelto. Llama a solve() primero.")
            return

        print("=" * 60)
        print(f"CIRCUITO TRIFÁSICO CA  –  {self.connection}  –  f={self.freq} Hz")
        print(f"  V_línea = {self.v_line} V   |   V_fase = {self.v_phase:.2f} V")
        print(f"  Carga:  {self.load['element']:10s}  {self.load['string']:>12}")
        print(f"  Z_carga = {self.load['impedance']:.4f} Ω")
        print("=" * 60)

        phases = ['A', 'B', 'C']
        if self.connection == 'Y':
            for ph in phases:
                r = self.results[ph]
                V, I = r['V_phase'], r['I_line']
                print(f"  Fase {ph}: |V|={abs(V):.2f} V ∠{np.degrees(np.angle(V)):.1f}°"
                      f"  |I|={abs(I):.4f} A ∠{np.degrees(np.angle(I)):.1f}°"
                      f"  P={r['P']:.2f} W  Q={r['Q']:.2f} VAr")
        else:
            for key, r in self.results.get('delta', {}).items():
                V, I = r['V_delta'], r['I_phase']
                print(f"  Δ {key}: |V_LL|={abs(V):.2f} V"
                      f"  |I_fase|={abs(I):.4f} A"
                      f"  P={r['P']:.2f} W  Q={r['Q']:.2f} VAr")
            for ph in phases:
                I_L = self.results[ph]['I_line']
                print(f"  Línea {ph}: |I_línea|={abs(I_L):.4f} A"
                      f"  ∠{np.degrees(np.angle(I_L)):.1f}°")

        print()
        print(f"  P_total = {self.P_total:.2f} W")
        print(f"  Q_total = {self.Q_total:.2f} VAr")
        print(f"  S_total = {self.S_total:.2f} VA")
        print(f"  Factor de potencia = {self.pf:.4f}")
        print()

    # ── Diagrama fasorial ─────────────────────────────────────────────────────

    def draw_phasors(self):
        """Dibuja el diagrama fasorial de tensiones de fase."""
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        colors = {'A': 'red', 'B': 'blue', 'C': 'green'}
        for ph, color in colors.items():
            V = self._v_phase_phasor(ph)
            ax.annotate('', xy=(np.angle(V), abs(V)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
            ax.text(np.angle(V), abs(V) * 1.1, f'V_{ph}', color=color, fontsize=12)

        ax.set_title(f"Diagrama fasorial – {self.connection}  V_fase={self.v_phase:.1f} V",
                     pad=20)
        plt.tight_layout()
        plt.show()
