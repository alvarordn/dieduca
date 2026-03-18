import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { CircuitosService } from '../../services/circuitos.service';
import { CircuitViewerComponent } from '../circuit-viewer/circuit-viewer.component';
import Swal from 'sweetalert2';

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
  public circuitoGenerado: any = null;
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

  limpiarEstado() {
    this.circuitoGenerado = null;
    this.preguntasEjemplo = [];
    this.resultadoVisible = false;
  }

  limpiarRespuestas() {
    this.resultadoVisible = false;
    this.mensajeResultado = '';
    this.preguntasEjemplo.forEach((p) => {
      p.respuestaUsuario = null;
      p.acertada = undefined;
    });
  }

  // Procesa el string complejo "real+imagj" y devuelve la magnitud (módulo)
  obtenerValorNumerico(val: any): number {
    if (!val) return 0;
    const str = val.toString().replace('j', '');
    const partes = str.match(/[+-]?\d+(\.\d+)?([eE][+-]?\d+)?/g) || [];
    const real = Number(partes[0] || 0);
    const imag = Number(partes[1] || 0);
    return Math.sqrt(real * real + imag * imag);
  }

  prepararPreguntas(circuito: any) {
    this.preguntasEjemplo = [];

    // 1. TENSIÓN DE NODO (Potential en cir.G.nodes)
    const nodosValidos = circuito.nodos.filter((n: any) => n.id !== 'N00');
    if (nodosValidos.length > 0) {
      const n = nodosValidos[Math.floor(Math.random() * nodosValidos.length)];
      console.log('nodo random', n);
      this.preguntasEjemplo.push({
        id: 'potencial',
        label: `Potencial eléctrico en el nodo ${n.id}`,
        valorReal: this.obtenerValorNumerico(n.potential),
        unidad: 'V',
        respuestaUsuario: null,
        acertada: undefined,
      });
    }

    // 2. CORRIENTE Y CAÍDA DE TENSIÓN (Current y V_drop en cir.G.edges)
    // Filtramos componentes reales (R, V, I) ignorando cables y esquinas
    const comps = circuito.componentes.filter(
      (c: any) => c.type !== 'wire' && c.type !== 'corner' && c.id,
    );
    const aleatorios = [...comps].sort(() => Math.random() - 0.5);

    // Preguntamos por corriente de uno
    if (aleatorios[0]) {
      this.preguntasEjemplo.push({
        id: 'corriente',
        label: `Intensidad de corriente en ${aleatorios[0].id}`,
        valorReal: this.obtenerValorNumerico(aleatorios[0].current),
        unidad: 'A',
        respuestaUsuario: null,
        acertada: undefined,
      });
    }

    // Preguntamos por caída de tensión de otro
    if (aleatorios[1]) {
      this.preguntasEjemplo.push({
        id: 'vdrop',
        label: `Caída de tensión (V_drop) en ${aleatorios[1].id}`,
        valorReal: this.obtenerValorNumerico(aleatorios[1].v_drop),
        unidad: 'V',
        respuestaUsuario: null,
        acertada: undefined,
      });
      console.log('Estas son preguntas', this.preguntasEjemplo);
    }

    // 3. DIFERENCIA DE POTENCIAL Vab (Entre dos nodos aleatorios)
    if (nodosValidos.length >= 2) {
      const nA = nodosValidos[0];
      const nB = nodosValidos[1];
      const vab = Math.abs(
        this.obtenerValorNumerico(nA.potential) -
          this.obtenerValorNumerico(nB.potential),
      );
      this.preguntasEjemplo.push({
        id: 'vab',
        label: `Diferencia de potencial Vab entre ${nA.id} y ${nB.id}`,
        valorReal: vab,
        unidad: 'V',
        respuestaUsuario: null,
        acertada: undefined,
      });
    }
  }

  comprobarRespuestas() {
    const respondidas = this.preguntasEjemplo.filter(
      (p) => p.respuestaUsuario !== null,
    ).length;

    if (respondidas < this.preguntasEjemplo.length) {
      this.mensajeResultado = 'Por favor, completa todas las preguntas.';
      this.resultadoVisible = true;
      return;
    }
 
    Swal.fire({
      title: '¿Enviar respuestas?',
      text: '',
      icon: 'question',
      showCancelButton: true,
      confirmButtonText: 'Sí, corregir',
      cancelButtonText: 'Revisar',
      confirmButtonColor: '#3085d6',
    }).then((result) => {
      if (result.isConfirmed) {
        let aciertos = 0;
        let fallos = 0;
        let totalPreguntas = 0;
        const tolerancia = 0.05;

        this.preguntasEjemplo.forEach((p) => {
          const real = Math.abs(p.valorReal);
          const usuario = Math.abs(p.respuestaUsuario);

          if (real < 0.0001) {
            p.acertada = usuario < 0.01;
          } else {
            const errorRelativo = Math.abs(real - usuario) / real;
            p.acertada = errorRelativo <= tolerancia;
          }
          if (p.acertada) {
            aciertos++;
            totalPreguntas++;
          } else {
            fallos++;
            totalPreguntas++;
          }

        });

        
          const aciertosPrevios = +(localStorage.getItem('Aciertos') || 0);
          const fallosPrevios = +(localStorage.getItem('Fallos') || 0);
          const totalPrevio = +(localStorage.getItem('TotalPreguntas') || 0);

          localStorage.setItem(
            'Aciertos',
            (aciertosPrevios + aciertos).toString(),
          );
          localStorage.setItem(
            'Fallos',
            (fallosPrevios + fallos).toString(),
          );
          localStorage.setItem(
            'TotalPreguntas',
            (totalPrevio + totalPreguntas).toString(),
          );

        const total = this.preguntasEjemplo.length;
        const id = this.bloqueId;

        const bAciertos = +(localStorage.getItem(`Aciertos_B${this.bloqueId}`) || 0);
        const bFallos = +(localStorage.getItem(`Fallos_B${this.bloqueId}`) || 0);
        const bTotal = +(localStorage.getItem(`Total_B${this.bloqueId}`) || 0);

        localStorage.setItem(`Aciertos_B${this.bloqueId}`, (bAciertos + aciertos).toString());
        localStorage.setItem(`Fallos_B${this.bloqueId}`, (bFallos + fallos).toString());
        localStorage.setItem(`Total_B${this.bloqueId}`, (bTotal + totalPreguntas).toString());
        this.mensajeResultado =
          aciertos === total
            ? `¡Excelente! ${aciertos}/${total} correctas.`
            : `Has acertado ${aciertos} de ${total}. ¡Sigue intentándolo!`;
        this.resultadoVisible = true;
      }
    });
  }

  generarEjercicio() {
    this.cargando = true;
    this.errorCircuito = '';
    const payload = { bloque: this.bloqueId, rows: this.rows, cols: this.cols };

    if (this.rows > 6 || this.cols > 6) {
      this.errorCircuito = 'Las filas/columnas tienen que ser menores q 7';
      this.cargando = false;
      this.circuitoGenerado = false;
      return;
    }

    if (this.rows < 2 || this.cols < 2) {
      this.errorCircuito = 'Las filas/columnas tienen que ser mayores a 1';
      this.cargando = false;
      this.circuitoGenerado = false;
      return;
    }

    this.circuitosService.generarCircuito(payload).subscribe({
      next: (res) => {
        this.cargando = false;
        if (res.success) {
          this.circuitoGenerado = res;
          this.prepararPreguntas(res.circuito);
        } else {
          this.errorCircuito = 'El circuito generado no es válido. Reintenta.';
        }
      },
      error: () => {
        this.cargando = false;
        this.errorCircuito = 'Error. Inicia sesion de nuevo';
      },
    });
  }
}
