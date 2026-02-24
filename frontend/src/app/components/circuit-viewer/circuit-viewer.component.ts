import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

// Representa un nodo del circuito dentro de una grilla.
// Cada nodo tiene una posición (row, col) y un tipo visual.
interface Nodo {
  id: string;      // Identificador único del nodo
  row: number;     // Fila dentro de la grilla
  col: number;     // Columna dentro de la grilla
  type: string;    // Tipo visual (esquina, borde, centro, etc.)
}

// Representa un componente eléctrico que conecta dos nodos.
interface Componente {
  id: string;             // Identificador único
  source: string;         // ID del nodo origen
  target: string;         // ID del nodo destino
  type: string;           // Tipo de componente (resistor, capacitor, etc.)
  value: string | null;   // Valor del componente (ej: 10Ω)
  orientation: string;    // 'horizontal' o 'vertical'
  labelPosition: string;  // Posición donde se dibuja el valor
}

@Component({
  selector: 'app-circuit-viewer', // Nombre para usar el componente en HTML
  standalone: true,               // No necesita módulo
  imports: [CommonModule],        // Importa directivas básicas (*ngIf, *ngFor)
  templateUrl: './circuit-viewer.component.html',
  styleUrl: './circuit-viewer.component.css'
})
export class CircuitViewerComponent implements OnChanges {

  // Datos que recibe desde el componente padre
  @Input() circuitData: any;

  // Constantes de diseño (controlan el tamaño del dibujo)
  public readonly CELL_SIZE = 200;   // Tamaño de cada celda de la grilla
  public readonly NODE_RADIUS = 18;  // Radio visual del nodo
  public readonly MARGIN = 80;       // Margen externo del dibujo

  // Dimensiones totales del lienzo
  imgWidth = 0;
  imgHeight = 0;

  // Listas internas procesadas
  nodos: Nodo[] = [];
  componentes: Componente[] = [];

  // Se ejecuta automáticamente cuando cambia algún @Input()
  ngOnChanges(changes: SimpleChanges) {
    // Si cambió circuitData y tiene valor válido
    if (changes['circuitData'] && this.circuitData) {
      this.processCircuitData();
    }
  }

  // Procesa los datos recibidos y calcula dimensiones del circuito
  processCircuitData() {

    // Carga nodos y componentes o usa arreglo vacío si no existen
    this.nodos = this.circuitData.nodos || [];
    this.componentes = this.circuitData.componentes || [];

    // Obtiene cantidad de filas y columnas (valores por defecto si no vienen)
    const rows = this.circuitData.rows || 2;
    const cols = this.circuitData.cols || 3;

    // Ajusta margen dinámicamente si el circuito es grande
    const marginX = this.MARGIN + (cols > 4 ? 20 : 0);
    const marginY = this.MARGIN + (rows > 3 ? 20 : 0);

    // Calcula ancho y alto total del área de dibujo
    this.imgWidth = (cols + 1) * this.CELL_SIZE + marginX * 2;
    this.imgHeight = (rows + 1) * this.CELL_SIZE + marginY * 2;
  }

  // Convierte una columna de la grilla en coordenada X (en píxeles)
  getNodeX(col: number): number {
    return this.MARGIN + (col + 0.5) * this.CELL_SIZE;
  }

  // Convierte una fila de la grilla en coordenada Y (en píxeles)
  getNodeY(row: number): number {
    return this.MARGIN + (row + 0.5) * this.CELL_SIZE;
  }

  // Busca un nodo por su ID
  getNode(nodeId: string): Nodo | undefined {
    return this.nodos.find(n => n.id === nodeId);
  }

  // Calcula el punto medio entre el nodo origen y destino
  // Sirve para colocar el componente visualmente entre ambos
  getComponentMidPoint(comp: Componente): { x: number, y: number } {

    const sourceNode = this.getNode(comp.source);
    const targetNode = this.getNode(comp.target);

    // Si alguno no existe, retorna posición por defecto
    if (!sourceNode || !targetNode) return { x: 0, y: 0 };

    return {
      x: (this.getNodeX(sourceNode.col) + this.getNodeX(targetNode.col)) / 2,
      y: (this.getNodeY(sourceNode.row) + this.getNodeY(targetNode.row)) / 2
    };
  }

  // Devuelve la rotación en grados según orientación
  getComponentRotation(comp: Componente): number {
    return comp.orientation === 'vertical' ? 90 : 0;
  }

  // Calcula la posición donde se dibujará el texto (valor del componente)
  getLabelPosition(comp: Componente): { x: number, y: number } {

    const mid = this.getComponentMidPoint(comp);
    const offset = 40; // Separación respecto al centro

    switch (comp.labelPosition) {
      case 'outside-top':
        return { x: mid.x, y: mid.y - offset - 20 };

      case 'outside-bottom':
        return { x: mid.x, y: mid.y + offset + 20 };

      case 'outside-left':
        return { x: mid.x - offset - 30, y: mid.y };

      case 'outside-right':
        return { x: mid.x + offset + 30, y: mid.y };

      case 'inside-bottom':
        return { x: mid.x, y: mid.y + offset };

      case 'inside-right':
        return { x: mid.x + offset, y: mid.y };

      default:
        return { x: mid.x + offset, y: mid.y };
    }
  }

  // Devuelve la ruta de la imagen según tipo de componente
  getComponentImgPath(type: string): string {

    const paths: Record<string, string> = {
      'resistor': 'assets/components/resistor.png',
      'capacitor': 'assets/components/capacitor.png',
      'inductor': 'assets/components/inductor.png',
      'v_source': 'assets/components/v_source.png',
      'c_source': 'assets/components/c_source.png',
      'shortcircuit': 'assets/components/shortcircuit.png',
      'opencircuit': 'assets/components/opencircuit.png'
    };

    // Si no encuentra el tipo, usa resistor por defecto
    return paths[type] || paths['resistor'];
  }

  // Devuelve la imagen correspondiente al tipo de nodo
  getNodeImgPath(type: string): string {

    const paths: Record<string, string> = {
      'corner-top-left': 'assets/nodes/corner-top-left.png',
      'corner-top-right': 'assets/nodes/corner-top-right.png',
      'corner-bottom-left': 'assets/nodes/corner-bottom-left.png',
      'corner-bottom-right': 'assets/nodes/corner-bottom-right.png',
      'edge-top': 'assets/nodes/edge-top.png',
      'edge-bottom': 'assets/nodes/edge-bottom.png',
      'edge-left': 'assets/nodes/edge-left.png',
      'edge-right': 'assets/nodes/edge-right.png',
      'center': 'assets/nodes/center.png'
    };

    // Si no coincide, usa nodo central por defecto
    return paths[type] || paths['center'];
  }
}