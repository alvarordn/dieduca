import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.optimize import fsolve


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


passive = ['capacitor',
           'inductor',
           'resistor']
active = ['v_source',
          'c_source']
limits = {'capacitor':  element_values(1e-9, 1e-3),
          'inductor':   element_values(1e-4, 1e-3),
          'resistor':   element_values(1e-2, 1e+2),
          'v_source':   element_values(1e+2, 1e+4),
          'c_source':   element_values(1e-1, 1e+2)}




class circuit():
    def __init__(self, dim, seed=None):
        if seed is not None:
            random.seed(seed)
        self.G = nx.DiGraph()
        self.nodes = [f"N{i}" for i in range(dim)]
        self.G.add_nodes_from(self.nodes)

        for _ in range(dim - 1):
            n1, n2 = random.sample(self.nodes, 2)
            if not self.G.has_edge(n1, n2):
                el = random.choice(passive)     
                val = random.choice(limits[el])
                if el == 'resistor':
                    imp = val                    
                if el == 'inductor':
                    imp = complex(0, 2*np.pi*50*val)
                if el == 'capacitor':
                    imp = complex(0, -1/(2*np.pi*50*val))           
                self.G.add_edge(n1, n2, element=el, value=val, impedance=imp)
        
        flag = True
        while flag:
            count = 0
            for i in range(dim):
                connections = list(set(self.G.successors(f"N{i}")) | set(self.G.predecessors(f"N{i}")))
                if len(connections) < 2:
                    neighs = self.nodes.copy()
                    neighs.remove(f"N{i}")
                    for item in connections:
                        neighs.remove(item)
                    el = random.choice(passive)  
                    val = random.choice(limits[el])
                    if el == 'resistor':
                        imp = val                    
                    if el == 'inductor':
                        imp = complex(0, 2*np.pi*50*val)
                    if el == 'capacitor':
                        imp = complex(0, -1/(2*np.pi*50*val))
                    self.G.add_edge(f"N{i}", random.choice(neighs), element=el, value=val, impedance=imp)
                else:
                    count += 1
            if count == dim:
                flag = False
                    
        self.edges = [(n1, n2, data.get('element')) for n1, n2, data in self.G.edges(data=True)]
        sources = np.max([np.floor(len(self.edges)/3), 1])
        edges = random.sample(self.edges, int(sources))
        for e in edges:
            self.G[e[0]][e[1]]["element"] = random.choice(active)
            self.G[e[0]][e[1]]["value"] = random.choice(limits[self.G[e[0]][e[1]]["element"]])
            
            
        self.edges = [(n1, n2, data.get('element'), data.get('value'), '1 mH') for n1, n2, data in self.G.edges(data=True)]
        
    def draw(self):
        pos = nx.kamada_kawai_layout(self.G)  # distribución de nodos
        edge_labels = nx.get_edge_attributes(self.G, 'element')

        plt.figure(figsize=(8, 6))
        nx.draw(
            self.G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=800,
            font_size=10,
            arrows=True,
            # connectionstyle="arc3,rad=0.15",
            arrowstyle='-|>',
            arrowsize=20
        )
        nx.draw_networkx_edge_labels(
            self.G, pos, edge_labels=edge_labels, font_color='darkred'
        )

    def solve(self):
        self.n = len(self.nodes)
        self.x = np.ones(self.n*2)
        sol = fsolve(self.iterate, self.x)
        return sol
        
    def iterate(self, x):
        self.set_voltages(x)
        self.compute_currents()
        res = self.compute_res()
        return res
        
    def set_voltages(self, x):
        idx = 0
        for node in self.nodes:
            self.G.nodes[node]["voltage"] = complex(x[0], x[1])
            idx += 2
        
    def compute_currents(self):
        for e in self.edges:
            if (self.G[e[0]][e[1]]["element"] == 'capacitor') or (self.G[e[0]][e[1]]["element"] == 'inductor') or (self.G[e[0]][e[1]]["element"] == 'resistor'):
                self.G[e[0]][e[1]]["current"] = (self.G.nodes[e[0]]["voltage"] - self.G.nodes[e[1]]["voltage"])/self.G[e[0]][e[1]]["impedance"]
            if self.G[e[0]][e[1]]["element"] == 'c_source':
                self.G[e[0]][e[1]]["current"] = self.G[e[0]][e[1]]["value"]        
        
        self.exclude = []
        for e in self.edges:
            if self.G[e[0]][e[1]]["element"] == 'v_source':
                node = e[1]
                self.exclude.append(node)
                edges = list(self.G.in_edges(node)) 
                total = 0
                try:
                    total += self.G[edges[0]][edges[1]]["current"]
                except:
                    pass
                edges = list(self.G.out_edges(node))
                try:
                    total -= self.G[edges[0]][edges[1]]["current"]
                except:
                    pass
                self.G[e[0]][e[1]]["current"] = total
        
    def compute_res(self):
        residuals = []
        for node in self.nodes:
            if node not in self.exclude:
                edges_in = list(self.G.in_edges(node)) 
                edges_out = list(self.G.out_edges(node))
                res = 0
                for e in edges_in:
                    res += self.G[e[0]][e[1]]["current"] 
                for e in edges_out:
                    res -= self.G[e[0]][e[1]]["current"] 
                residuals.append(np.real(res))
                residuals.append(np.imag(res))
        for e in self.edges:
            if self.G[e[0]][e[1]]["element"] == 'v_source':
                res = self.G.nodes[e[0]]["voltage"] - self.G.nodes[e[1]]["voltage"] - self.G[e[0]][e[1]]["value"]
                residuals.append(np.real(res))
                residuals.append(np.imag(res))
        return residuals
        
        
        
        
        
        
        


