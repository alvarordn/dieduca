import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { CircuitosService } from '../../services/circuitos.service';
import { CircuitViewerComponent } from '../circuit-viewer/circuit-viewer.component';

@Component({
  selector: 'app-bloque',
  standalone: true,
  imports: [CommonModule, FormsModule, CircuitViewerComponent],
  templateUrl: './bloque.component.html',
  styleUrl: './bloque.component.css',
})
export class BloqueComponent implements OnInit {
  public bloqueId: number = 0;
  public cargando: boolean = false;
  public errorCircuito: string = '';
  public circuitoGenerado = <any>{};
  public mensajeResultado: string = '';
  public resultadoVisible: boolean = false;

  preguntasEjemplo: any[] = [];

  rows: number = 2;
  cols: number = 3;

  constructor(
    private route: ActivatedRoute,
    private circuitosService: CircuitosService,
  ) {}

  ngOnInit() {
    this.route.params.subscribe((params) => {
      this.bloqueId = params['id'];
      this.limpiarEstado();
    });
  }

  limpiarRespuestas() {
    this.resultadoVisible = false;
    this.mensajeResultado = '';
    this.resultadoVisible = false;
    this.mensajeResultado = '';
    this.errorCircuito = '';
    this.preguntasEjemplo.forEach((p) => {
      p.respuestaUsuario = null; // Borra el número escrito
      p.acertada = undefined; // <--- ESTO QUITA EL COLOR (verde/rojo)
    });
  }

  limpiarEstado() {}

  calcularMagnitud(seleccion: any[]) {
    console.log('Calcular magnitud', seleccion);

    this.preguntasEjemplo = seleccion.map((n) => {
      //quitamos la j

      const sinJ = n.potential.replace('j', '');
      console.log('sin j', sinJ);

      const partes = sinJ.match(/[+-]?\d+(\.\d+)?/g) || [];

      console.log('partes', partes);

      const real = Number(partes[0] || 0);
      console.log('Real', real);

      const imag = Number(partes[1] || 0);
      console.log('imaginario', imag);

      const magnitud = Math.sqrt(real * real + imag * imag);
      console.log('magnitud', magnitud);

      return {
        label: `Magnitud de tensión en ${n.id}`,
        valorReal: magnitud,
        respuestaUsuario: null,
      };
    });
  }

  comprobarRespuestas() {
    const TOLERANCIA = 0.01; // tolerancia fija o ajustable
    let aciertos = 0;

    this.preguntasEjemplo.forEach((p) => {
      if (p.respuestaUsuario != null) {
        const diferencia = Math.abs(p.respuestaUsuario - p.valorReal);
        p.acertada = diferencia <= TOLERANCIA; // true si acierta, false si no
        if (p.acertada) aciertos++;
        const total = this.preguntasEjemplo.length;
        this.mensajeResultado =
          aciertos === total
            ? `¡Excelente! Has acertado todas (${aciertos}/${total}).`
            : `Has acertado ${aciertos} de ${total}. Revisa tus cálculos.`;

        this.resultadoVisible = true;
      } else {
        this.mensajeResultado = 'Rellena todos los campos';
        this.resultadoVisible = true;
      }
    });
  }

  // bloque.component.ts

  formatearEtiqueta(texto: string): string {
    if (!texto) return '';

    return texto
      .replace(/-/g, '') // Quita todos los guiones
      .replace(/micro/g, 'μ') // Cambia "micro" por "μ"
      .replace(/micro-/g, 'μ') // Por si acaso viene con guion
      .replace(/n-/g, 'n') // Limpia nano-faradios si aparecen
      .replace(/m-/g, 'm'); // Limpia mili-ohmios (ej: 30 m-Ω -> 30 mΩ)
  }

  prepararPreguntas(nodos: any[]) {
    console.log('nodos', nodos);

    const nodosPosibles = nodos.filter((n) => n.id !== 'N00');
    console.log('Nodos posibles', nodosPosibles);

    const aleatorios = nodosPosibles.sort(() => Math.random() - 0.5);

    const seleccion = aleatorios.slice(0, 4);
    console.log('seleccion', seleccion);

    this.preguntasEjemplo = seleccion.map((n) => ({
      label: `Magnitud de tensión en  ${n.id}`,
      respuestaUsuario: null,
    }));
    this.calcularMagnitud(seleccion);
  }

  generarEjercicio() {
    const datos = {
      bloque: this.bloqueId,
      rows: Number(this.rows),
      cols: Number(this.cols),
    };
    console.log('Ejercicio Generado');
    this.circuitosService.generarCircuito(datos).subscribe({
      next: (datos) => {
        console.log('Estos son los datos', datos);
        this.circuitoGenerado = datos;

        datos.circuito.componentes.forEach((comp: any) => {
          if (comp.value) {
            comp.value = comp.value
              .replace(/-/g, '')
              .replace(/micro/g, 'μ')
              .replace(/m-Ω/g, 'mΩ');
          }
        });

        this.prepararPreguntas(datos.circuito.nodos);
      },
    });
  }
}
