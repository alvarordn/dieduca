import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

interface Nodo {
  id: string;
  row: number;
  col: number;
  type: string;
}

interface Componente {
  id: string;
  source: string;
  target: string;
  type: string;
  value: string | null;
  orientation: string;
  labelPosition: string;
}

@Component({
  selector: 'app-circuit4-view',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './circuit4-view.component.html',
  styleUrl: './circuit4-view.component.css',
})
export class Circuit4ViewComponent implements OnChanges {
  // Recibe la respuesta del backend (que contiene exámen, filas, columnas, etc.)
  @Input() circuitData: any = null;

  // Parámetros de dimensionamiento espacial SVG
  public readonly CELL_SIZE = 240;
  public readonly MARGIN = 120;

  imgWidth = 0;
  imgHeight = 0;

  nodos: Nodo[] = [];
  componentes: Componente[] = [];

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['circuitData'] && this.circuitData) {
      this.procesarTopologia();
    }
  }

  private procesarTopologia(): void {
    // Si viene anidador dentro de 'circuito', lo extraemos directamente
    const rawData = this.circuitData.circuito
      ? this.circuitData.circuito
      : this.circuitData;

    this.nodos = rawData.nodos || [];
    this.componentes = rawData.componentes || [];

    const rows = rawData.rows || 4;
    const cols = rawData.cols || 5;

    // Establecer ancho y alto adaptativo de la lona virtual SVG
    this.imgWidth = (cols - 1) * this.CELL_SIZE + this.MARGIN * 2;
    this.imgHeight = (rows - 1) * this.CELL_SIZE + this.MARGIN * 2;
  }

  public getNodeX(col: number): number {
    return this.MARGIN + col * this.CELL_SIZE;
  }

  public getNodeY(row: number): number {
    return this.MARGIN + row * this.CELL_SIZE;
  }

  public getNode(nodeId: string): Nodo | undefined {
    return this.nodos.find((n) => n.id === nodeId);
  }

  public getComponentMidPoint(comp: Componente): { x: number; y: number } {
    const sourceNode = this.getNode(comp.source);
    const targetNode = this.getNode(comp.target);

    if (!sourceNode || !targetNode) return { x: 0, y: 0 };

    return {
      x: (this.getNodeX(sourceNode.col) + this.getNodeX(targetNode.col)) / 2,
      y: (this.getNodeY(sourceNode.row) + this.getNodeY(targetNode.row)) / 2,
    };
  }

  // Rotación idéntica al estándar trifásico
  public getComponentRotation(comp: Componente): number {
    return comp.orientation === 'vertical' ? 0 : 90;
  }

  // Posicionamiento inteligente del texto para que no colisione con las líneas
  public getLabelPosition(comp: Componente): { x: number; y: number } {
    const mid = this.getComponentMidPoint(comp);
    const offset = 55;

    switch (comp.labelPosition) {
      case 'outside-top':
        return { x: mid.x, y: mid.y - offset };
      case 'outside-bottom':
        return { x: mid.x, y: mid.y + offset };
      case 'outside-left':
        return { x: mid.x - offset - 15, y: mid.y };
      case 'outside-right':
        return { x: mid.x + offset + 15, y: mid.y };
      case 'inside-right':
        return { x: mid.x + 30, y: mid.y - 20 };
      default:
        return { x: mid.x, y: mid.y - offset };
    }
  }

  public getComponentImgPath(type: string): string {
    // Si el tipo es 'c_source', busca exactamente 'c_source.png'
    // Si tienes otros componentes, se mantendrá su nombre original
    return `assets/components/${type}.png`;
  }

  // Convierte IDs técnicos en las etiquetas literales de los exámenes
  public getNudoLetra(id: string): string {
    if (id === 'N10' || id === 'N20') return 'A';
    if (id === 'N12' || id === 'N22') return 'B';
    if (id === 'N24') return 'C';
    return '';
  }

  public formatearEtiqueta(valor: string | null): string {
    if (!valor) return '';
    return valor
      .replace(/micro/g, 'μ')
      .replace(/-/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }
}
