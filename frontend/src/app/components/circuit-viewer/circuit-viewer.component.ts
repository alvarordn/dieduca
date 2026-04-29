import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

// estructura de un nodo del circuito (puntos de la grilla)
interface Nodo {
  id: string; // id único del nodo
  row: number; // fila en la rejilla
  col: number; // columna en la rejilla
  type: string; // tipo visual del nodo (esquina, centro, etc.)
}

// estructura de un componente eléctrico (resistencia, etc.)
interface Componente {
  id: string; // id único del componente
  source: string; // nodo origen
  target: string; // nodo destino
  type: string; // tipo de componente (resistor, etc.)
  value: string | null; // valor del componente (ej: 10Ω)
  orientation: string; // orientación horizontal o vertical
  labelPosition: string; // dónde se dibuja el texto
}

@Component({
  selector: 'app-circuit-viewer',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './circuit-viewer.component.html',
  styleUrl: './circuit-viewer.component.css',
})
export class CircuitViewerComponent implements OnChanges {

  // datos que llegan desde el componente padre
  @Input() circuitData: any;

  // constantes para controlar el tamaño del dibujo
  public readonly CELL_SIZE = 200; // tamaño de cada celda del grid
  public readonly NODE_RADIUS = 18; // radio visual del nodo
  public readonly MARGIN = 100; // margen del circuito

  // tamaño total del SVG
  imgWidth = 0;
  imgHeight = 0;

  // listas internas ya procesadas
  nodos: Nodo[] = [];
  componentes: Componente[] = [];

  // se ejecuta cuando cambia el input (circuitData)
  ngOnChanges(changes: SimpleChanges) {
    if (changes['circuitData'] && this.circuitData) {
      this.processCircuitData(); // recalculo todo el circuito
    }
  }

  ngOnInit(){
    // debug para ver qué datos llegan
    console.log("Datos del circuito", this.circuitData)
  }

  // procesa los datos del circuito para poder dibujarlo
  processCircuitData() {

    // guardo nodos y componentes (si no hay, array vacío)
    this.nodos = this.circuitData.nodos || [];
    this.componentes = this.circuitData.componentes || [];

    console.log(this.circuitData.componentes)

    // tamaño de la rejilla (filas y columnas)
    const rows = this.circuitData.rows;
    const cols = this.circuitData.cols;

    // calculo tamaño total del SVG
    this.imgWidth = (cols - 1) * this.CELL_SIZE + this.MARGIN * 2;
    this.imgHeight = (rows - 1) * this.CELL_SIZE + this.MARGIN * 2;
  }

  // convierte columna de grid a coordenada X en píxeles
  getNodeX(col: number): number {
    return this.MARGIN + col * this.CELL_SIZE;
  }

  // convierte fila de grid a coordenada Y en píxeles
  getNodeY(row: number): number {
    return this.MARGIN + row * this.CELL_SIZE;
  }

  // busca un nodo por su id
  getNode(nodeId: string): Nodo | undefined {
    return this.nodos.find((n) => n.id === nodeId);
  }

  // calcula el punto medio entre dos nodos (para centrar componentes)
  getComponentMidPoint(comp: Componente): { x: number; y: number } {
    const sourceNode = this.getNode(comp.source);
    const targetNode = this.getNode(comp.target);

    // si falta alguno, devuelvo 0 por seguridad
    if (!sourceNode || !targetNode) return { x: 0, y: 0 };

    return {
      x: (this.getNodeX(sourceNode.col) + this.getNodeX(targetNode.col)) / 2,
      y: (this.getNodeY(sourceNode.row) + this.getNodeY(targetNode.row)) / 2,
    };
  }

  // decide cómo rotar el componente según orientación
  getComponentRotation(comp: Componente): number {
    return comp.orientation === 'vertical' ? 180 : 90;
  }

  // calcula dónde poner el texto del componente
  getLabelPosition(comp: Componente): { x: number; y: number } {
    const mid = this.getComponentMidPoint(comp);
    const offset = 40; // separación respecto al centro

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

      // posición por defecto si no coincide nada
      default:
        return { x: mid.x + offset, y: mid.y };
    }
  }

  // ruta de imagen según tipo de componente
  getComponentImgPath(type: string): string {
    return `assets/components/${type}.png`;
  }

  // ruta de imagen según tipo de nodo
  getNodeImgPath(type: string): string {
    return `assets/nodes/${type}.png`;
  }

  // formatea etiquetas para que se vean mejor
  formatearEtiqueta(valor: string | null): string {
    if (!valor) return '';

    return valor
      .replace(/micro/g, 'μ') // cambia "micro" por símbolo real
      .replace(/-/g, '')      // quita guiones
      .replace(/\s+/g, ' ')   // limpia espacios dobles
      .trim();                // quita espacios finales
  }
}