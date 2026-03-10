import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math

random.seed(1)

def most_significant_digit(num):
    exp = math.floor(math.log10(abs(num)))
    cifra = int(abs(num) / 10**exp)
    return -exp

msd = [(0, ' '),
       (3, 'm-'),
       (6, 'micro-'),
       (9, 'n-')]

def format_with_prefix(val):
    exp = most_significant_digit(val)

    for threshold, prefix in msd:
        if exp <= threshold:
            break

    scale = 10 ** threshold
    scaled_val = val * scale

    text = f"{scaled_val:g} {prefix}".strip()
    return text

def element_values(lb, ub):
    lb_exp = int(np.floor(np.log10(lb)))
    ub_exp = int(np.floor(np.log10(ub)))

    values = []
    for exp in range(lb_exp, ub_exp + 1):
        base = 10 ** exp
        for i in range(1, 10):
            val = i * base
            if lb <= val <= ub:
                values.append(val)
    return values


limits = {'capacitor':  element_values(1e-9, 1e-3),
          'inductor':   element_values(1e-4, 1e-3),
          'resistor':   element_values(1e-2, 1e+2),
          'v_source':   element_values(1e+2, 1e+4),
          'c_source':   element_values(1e-1, 1e+2)}
passive = ['capacitor',
           'inductor',
           'resistor',
           'shortcircuit',
           'opencircuit']
active = ['v_source',
          'c_source']

class Circuit:
    def __init__(self, rows, cols, freq=50, seed=None):
        if seed is not None:
            random.seed(seed)

        self.rows = rows
        self.cols = cols
        self.freq = freq
        self.omega = 2 * np.pi * freq
        self.G = nx.DiGraph()

        self.nodes = [f"N{r}{c}" for r in range(rows) for c in range(cols)]
        self.G.add_nodes_from(self.nodes)

        self._add_edges(rows, cols)

        self._inject_sources()

    def _get_element_data(self, el_type):
        if el_type in ['shortcircuit', 'opencircuit']:
            return None, None, el_type

        val = random.choice(limits[el_type])
        prefix_val = format_with_prefix(val)

        mapping = {
            'resistor':  (val, f"{prefix_val}Ω", val),
            'inductor':  (val, f"{prefix_val}H",  complex(0, self.omega * val)),
            'capacitor': (val, f"{prefix_val}F",  complex(0, -1 / (self.omega * val)))
        }
        return mapping.get(el_type)

    def _add_edges(self, rows, cols):
        for r in range(rows):
            for c in range(cols):
                if c < cols - 1:
                    self._create_edge(f"N{r}{c}", f"N{r}{c+1}")
                if r < rows - 1:
                    self._create_edge(f"N{r}{c}", f"N{r+1}{c}")

    def _create_edge(self, u, v):
        el = random.choice(passive)
        val, string_val, imp = self._get_element_data(el)
        self.G.add_edge(u, v, element=el, value=val, impedance=imp, string=string_val)

    def _inject_sources(self):
        num_sources = max(int((2 * self.rows * self.cols - self.rows - self.cols) // 5), 1)
        current_edges = list(self.G.edges())
        edges_to_replace = random.sample(current_edges, num_sources)

        for u, v in edges_to_replace:
            el = random.choice(active)
            val = random.choice(limits[el])
            unit = 'V' if el == 'v_source' else 'A'

            self.G[u][v].update({
                "element": el,
                "value": val,
                "impedance": None,
                "string": f"{val} {unit}"
            })

    def draw(self):
        pos = nx.planar_layout(self.G)
        edge_labels = nx.get_edge_attributes(self.G, 'element')
        plt.figure(figsize=(8, 6))
        nx.draw(
            self.G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=800,
            font_size=10,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20
        )
        nx.draw_networkx_edge_labels(
            self.G, pos, edge_labels=edge_labels, font_color='darkred'
        )
        plt.show()


    def solve(self):
        nodes = list(self.G.nodes())
        num_nodes = len(nodes)

        v_sources = [e for e in self.G.edges(data=True) if e[2]['element'] in ['v_source', 'shortcircuit']]
        num_v = len(v_sources)

        size = (num_nodes - 1) + num_v
        A = np.zeros((size, size), dtype=complex)
        Z = np.zeros(size, dtype=complex)

        # El primer nodo en 'nodes' se asume como Tierra (Potencial 0)
        node_map = {node: i-1 for i, node in enumerate(nodes) if i > 0}

        for u, v, data in self.G.edges(data=True):
            elem = data['element']
            # Impedances
            if elem not in ['v_source', 'c_source', 'shortcircuit', 'opencircuit']:
                y = 1.0 / data['impedance']
                if u in node_map:
                    A[node_map[u], node_map[u]] += y
                if v in node_map:
                    A[node_map[v], node_map[v]] += y
                if u in node_map and v in node_map:
                    A[node_map[u], node_map[v]] -= y
                    A[node_map[v], node_map[u]] -= y

            # Current sources
            elif elem == 'c_source':
                val = data['value']
                if u in node_map: Z[node_map[u]] -= val
                if v in node_map: Z[node_map[v]] += val

        # Voltage sources and shortcircuits
        for i, (u, v, data) in enumerate(v_sources):
            idx_v = (num_nodes - 1) + i
            val = data['value'] if data['element'] == 'v_source' else 0j
            if u in node_map:
                A[idx_v, node_map[u]] = -1
                A[node_map[u], idx_v] = -1
            if v in node_map:
                A[idx_v, node_map[v]] = 1
                A[node_map[v], idx_v] = 1
            Z[idx_v] = val

        # Solving
        try:
            X = np.linalg.solve(A, Z)
        except np.linalg.LinAlgError:
            print("Error: El circuito es singular.")
            return False

        node_potentials = {node: (X[node_map[node]] if node in node_map else 0j) for node in nodes}
        for n in self.G.nodes():
            self.G.nodes[n]['potential'] = node_potentials[n]
        for u, v, data in self.G.edges(data=True):
            v_diff = node_potentials[u] - node_potentials[v]
            data['v_drop'] = v_diff

            if data['element'] in ['v_source', 'shortcircuit']:
                for idx, (su, sv, sd) in enumerate(v_sources):
                    if (u, v) == (su, sv) and data is sd:
                        data['current'] = X[(num_nodes - 1) + idx]
                        break
            elif data['element'] == 'c_source':
                data['current'] = data['value']
            elif data['element'] == 'opencircuit':
                data['current'] = 0j
            else:
                data['current'] = v_diff / data['impedance']
        return True
