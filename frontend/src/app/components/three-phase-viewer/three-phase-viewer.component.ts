import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

// Tipo para controlar las fases del sistema trifásico
type Fase = 'A' | 'B' | 'C';

@Component({
  selector: 'app-three-phase-viewer',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './three-phase-viewer.component.html',
  styleUrl: './three-phase-viewer.component.css',
})
export class ThreePhaseViewerComponent  {

  // Datos que vienen desde el componente padre 
  @Input() data: any;

  mensajeError: string = 'No pueden ser mas de 5 secciones';

  // Array con las fases eléctricas del sistema
  fases: Fase[] = ['A', 'B', 'C'];

  // Posición vertical de cada fase en el SVG
  readonly FaseY: Record<Fase, number> = {
    A: 100,
    B: 260,
    C: 420,
  };

  // Colores asociados a cada fase (para visualización)
  readonly Color: Record<Fase, string> = {
    A: 'black', 
    B: 'black', 
    C: 'black', 
  };


  // Getter que asegura que siempre haya un circuito válido
  // (evita errores si data viene undefined)
  get circuit() {
    return this.data ?? { sections: [], params: {}, results: {} };
  }

  // Devuelve las secciones del circuito (o array vacío si no hay)
  get sections() {
    return this.circuit.sections || [];
  }

  // Devuelve parámetros del circuito (frecuencia, tensión, etc.)
  get params() {
    return this.circuit.params || {};
  }

  // Calcula la posición horizontal de cada sección en el SVG
  getSectionX(i: number) {
    return 220 + i * 260;
  }

  // Devuelve el color correspondiente a cada fase
  getColor(p: Fase) {
    return this.Color[p];
  }

  // Calcula el centro vertical entre fase A y C (para elementos centrales)
  centerY() {
    return (this.FaseY.A + this.FaseY.C) / 2;
  }

  // Calcula el final del circuito en eje X
  // (última sección + margen visual)
  getEndX() {
    const final = this.sections?.length - 1;
    return this.getSectionX(final) + 120;
  }

  
}