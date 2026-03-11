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
  bloqueId: string = '';
  circuitoGenerado: any = null;
  cargando: boolean = false;
  errorCircuito: string = '';

  // Configuración de la rejilla
  rows: number = 2;
  cols: number = 3;

  // Array dinámico de preguntas y objeto de respuestas
  preguntasEjemplo: any[] = [];
  respuestas: { [key: string]: number | null } = {};

  resultadoVisible: boolean = false;
  mensajeResultado: string = '';

  constructor(
    private route: ActivatedRoute,
    private circuitosService: CircuitosService
  ) {}

  ngOnInit() {
    this.route.params.subscribe((params) => {
      this.bloqueId = params['id'];
      this.limpiarEstado();
    });
  }

  /**
   * 1. Lógica para procesar el string complejo de Python
   * Ejemplo: "-0.01824-20.8929j" -> Calcula la magnitud sqrt(a² + b²)
   */
  calcularMagnitudComplejo(valor: string | number): number {
    if (typeof valor === 'number') return Math.abs(valor);
    if (!valor) return 0;

    // Limpiamos espacios y la 'j' final
    let s = valor.replace(/\s/g, '').replace(/j$/, '');

    // Extraemos las partes numérica (real e imaginaria)
    const matches = s.match(/[+-]?\d+(\.\d+)?/g);

    if (matches) {
      const real = parseFloat(matches[0]);
      const imag = matches.length > 1 ? parseFloat(matches[1]) : 0;
      
      // Retornamos el módulo del número complejo
      return Math.sqrt(Math.pow(real, 2) + Math.pow(imag, 2));
    }
    return 0;
  }

  /**
   * 2. Generador de preguntas basado en los datos reales del backend
   */
  prepararPreguntas(nodos: any[]) {
    // Filtramos N00 (tierra) porque siempre es 0V
    const nodosCandidatos = nodos.filter(n => n.id !== 'N00');

    // Barajamos y seleccionamos 4 nodos al azar
    const seleccion = nodosCandidatos
      .sort(() => 0.5 - Math.random())
      .slice(0, 4);

    this.preguntasEjemplo = seleccion.map((nodo, index) => {
      return {
        id: index,
        label: `Magnitud de tensión en el nodo ${nodo.id}`,
        unidad: 'V',
        valorReal: this.calcularMagnitudComplejo(nodo.potential)
      };
    });
  }

  generarEjercicio() {
    this.limpiarEstado();
    this.cargando = true;

    const datos = {
      bloque: this.bloqueId,
      rows: Number(this.rows),
      cols: Number(this.cols),
    };

    this.circuitosService.generarCircuito(datos).subscribe({
      next: (respuesta) => {
        this.cargando = false;
        if (respuesta && respuesta.success) {
          this.circuitoGenerado = respuesta;
          // Generamos las preguntas automáticas usando los nodos recibidos
          this.prepararPreguntas(respuesta.circuito.nodos);
        } else {
          this.errorCircuito = 'El circuito generado no es válido.';
        }
      },
      error: (err) => {
        this.cargando = false;
        this.errorCircuito = 'Error de conexión con el servidor.';
        console.error(err);
      }
    });
  }

  /**
   * 3. Comprobación de respuestas con margen de error (Tolerancia)
   */
  comprobarRespuestas() {
    let aciertos = 0;
    const TOLERANCIA = 0.05; // Margen de 0.05 unidades

    this.preguntasEjemplo.forEach((p, i) => {
      const respUsuario = this.respuestas['p' + i];
      
      if (respUsuario !== null && respUsuario !== undefined) {
        const diferencia = Math.abs(respUsuario - p.valorReal);
        if (diferencia <= TOLERANCIA) {
          aciertos++;
        }
      }
    });

    this.mensajeResultado = aciertos === this.preguntasEjemplo.length
      ? `¡Excelente! Has acertado todas (${aciertos}/${this.preguntasEjemplo.length}).`
      : `Has acertado ${aciertos} de ${this.preguntasEjemplo.length}. Revisa tus cálculos.`;
    
    this.resultadoVisible = true;
  }

  limpiarEstado() {
    this.circuitoGenerado = null;
    this.errorCircuito = '';
    this.respuestas = {};
    this.preguntasEjemplo = [];
    this.resultadoVisible = false;
    this.mensajeResultado = '';
  }

  limpiarRespuestas() {
    this.respuestas = {};
    this.resultadoVisible = false;
    this.mensajeResultado = '';
  }
}