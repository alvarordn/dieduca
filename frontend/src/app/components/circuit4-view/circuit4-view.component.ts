import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

// Nodo del circuito (cada punto de la rejilla)
interface Nodo {
  id: string; // identificador tipo N1, N2...
  row: number; // fila en la grid
  col: number; // columna en la grid
  type: string; // tipo visual (normal, tierra, etc.)
}

// Componente eléctrico entre dos nodos
interface Componente {
  id: string; // R1, V1, etc.
  source: string; // nodo inicio
  target: string; // nodo final
  type: string; // resistor, fuente, etc.
  value: string | null; // valor (ohmios, voltios...)
  orientation: string; // horizontal / vertical
  labelPosition?: string; // opcional
}

@Component({
  selector: 'app-circuit4-view',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './circuit4-view.component.html',
  styleUrl: './circuit4-view.component.css',
})
export class Circuit4ViewComponent implements OnChanges {
  // 👇 input que viene del padre con el circuito entero
  @Input() circuitData: any = null;

  @Input() numeroProblema!: number;

  // tamaño de cada “celda” del circuito (más pequeño = más compacto)
  readonly CELL_SIZE = 160;

  // margen exterior para que no quede pegado al borde
  readonly MARGIN = 60;

  // tamaño real del SVG
  imgWidth = 0;
  imgHeight = 0;

  // arrays donde guardo nodos y componentes ya procesados
  nodos: Nodo[] = [];
  componentes: Componente[] = [];

  // se ejecuta cuando cambia el input del circuito
  ngOnChanges(changes: SimpleChanges): void {
    // si llega nuevo circuito, lo recalculo todo
    if (changes['circuitData'] && this.circuitData) {
      this.procesarTopologia();
    }
  }

  // convierte el JSON del backend en nodos + componentes listos para dibujar
  private procesarTopologia(): void {
    // backend puede mandar circuito dentro de "circuito" o directo
    const rawData = this.circuitData?.circuito ?? this.circuitData;
    this.numeroProblema = this.circuitData.plantilla

    // saco nodos y componentes o vacío si no hay
    this.nodos = rawData?.nodos ?? [];
    this.componentes = rawData?.componentes ?? [];

    // dimensiones del grid
    const rows = rawData?.rows ?? 4;
    const cols = rawData?.cols ?? 4;

    // calculo tamaño total del SVG
    this.imgWidth = (cols - 1) * this.CELL_SIZE;
    this.imgHeight = (rows - 1) * this.CELL_SIZE;
  }

  // POSICIONES EN EL SVG

  // convierte columna a coordenada X real
  getNodeX(col: number): number {
    return col * this.CELL_SIZE + this.MARGIN;
  }

  // convierte fila a coordenada Y real
  getNodeY(row: number): number {
    return row * this.CELL_SIZE + this.MARGIN;
  }

  // busca un nodo por id
  getNode(nodeId: string): Nodo | undefined {
    return this.nodos.find((n) => n.id === nodeId);
  }

  // calcula el punto medio entre dos nodos (para dibujar componentes)
  getComponentMidPoint(comp: Componente) {
    const s = this.getNode(comp.source);
    const t = this.getNode(comp.target);

    if (!s || !t) return { x: 0, y: 0 };

    return {
      x: (this.getNodeX(s.col) + this.getNodeX(t.col)) / 2,
      y: (this.getNodeY(s.row) + this.getNodeY(t.row)) / 2,
    };
  }

  // rotación del componente según orientación
  getComponentRotation(comp: Componente): number {
    return comp.orientation === 'vertical' ? 0 : 90;
  }

  // posición del label (un poco encima del componente)
  getLabelPosition(comp: Componente) {
    const mid = this.getComponentMidPoint(comp);
    return { x: mid.x, y: mid.y - 35 };
  }

  // ruta de la imagen del componente
  getComponentImgPath(type: string): string {
    return `assets/components/${type}.png`;
  }

  // viewBox del SVG (para que se vea todo bien centrado)
  getViewBox(): string {
    const padding = this.MARGIN * 2;

    const width = this.imgWidth + padding;
    const height = this.imgHeight + padding;

    return `-${this.MARGIN} -${this.MARGIN} ${width} ${height}`;
  }
}
