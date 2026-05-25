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
  labelPosition?: string;
}

@Component({
  selector: 'app-circuit4-view',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './circuit4-view.component.html',
  styleUrl: './circuit4-view.component.css',
})
export class Circuit4ViewComponent implements OnChanges {

  @Input() circuitData: any = null;

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

    const rawData = this.circuitData?.circuito ?? this.circuitData;

    this.nodos = rawData?.nodos ?? [];
    this.componentes = rawData?.componentes ?? [];

    const rows = rawData?.rows ?? 4;
    const cols = rawData?.cols ?? 5;

    this.imgWidth = (cols - 1) * this.CELL_SIZE;
    this.imgHeight = (rows - 1) * this.CELL_SIZE;
  }

  // =========================
  // COORDENADAS
  // =========================
  public getNodeX(col: number): number {
    return col * this.CELL_SIZE;
  }

  public getNodeY(row: number): number {
    return row * this.CELL_SIZE;
  }

  public getNode(nodeId: string): Nodo | undefined {
    return this.nodos.find(n => n.id === nodeId);
  }

  public getComponentMidPoint(comp: Componente) {
    const s = this.getNode(comp.source);
    const t = this.getNode(comp.target);

    if (!s || !t) return { x: 0, y: 0 };

    return {
      x: (this.getNodeX(s.col) + this.getNodeX(t.col)) / 2,
      y: (this.getNodeY(s.row) + this.getNodeY(t.row)) / 2,
    };
  }

  public getComponentRotation(comp: Componente): number {
    return comp.orientation === 'vertical' ? 0 : 90;
  }

  public getLabelPosition(comp: Componente) {
    const mid = this.getComponentMidPoint(comp);
    return { x: mid.x, y: mid.y - 50 };
  }

  public getComponentImgPath(type: string): string {
    return `assets/components/${type}.png`;
  }
}