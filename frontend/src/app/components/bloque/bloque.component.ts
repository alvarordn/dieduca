import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { CircuitosService } from '../../services/circuitos.service';
import { CircuitViewerComponent } from '../circuit-viewer/circuit-viewer.component';
import { ThreePhaseViewerComponent } from '../three-phase-viewer/three-phase-viewer.component';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import Swal from 'sweetalert2';
import { Bloque4Component } from '../bloque4/bloque4.component';

@Component({
  selector: 'app-bloque',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    CircuitViewerComponent,
    ThreePhaseViewerComponent,
    Bloque4Component
  ],
  templateUrl: './bloque.component.html',
  styleUrl: './bloque.component.css',
})
export class BloqueComponent implements OnInit {
  // El ID que pillo de la URL para saber en qué bloque estamos
  public bloqueId = 0;

  // Los nombres de los temas para ponerlos en el título
  TEMAS_BLOQUES: { [key: number]: string } = {
    1: 'Conceptos fundamentales y leyes de Kirchhoff',
    2: 'Circuitos Resistivos con Generadores Ideales',
    3: 'Fuentes reales y circuitos equivalentes',
    4: 'Técnicas de análisis de circuitos',
    5: 'Componentes dinámicos',
    6: 'Análisis de circuitos de corriente continua en distintos regímenes temporales',
    7: 'Resolución de circuitos de corriente alterna sinusoidal',
    8: 'Potencia y energía en circuitos de corriente alterna sinusoidal',
    9: 'Circuitos trifásicos',
    10: 'Potencia en circuitos trifásicos equilibrados',
  };

  public cargando = false;
  public errorCircuito = '';
  public circuitoGenerado: any = null;
  public mensajeResultado = '';
  public resultadoVisible = false;
  tipoCircuito: 'monofasico' | 'trifasico' | null = null;
  preguntasEjemplo: any[] = [];

  rows = 2;
  cols = 3;
  public tituloBloque = '';
  secciones: number = 2;

  constructor(
    private route: ActivatedRoute,
    private circuitosService: CircuitosService,
    private http: HttpClient,
  ) {}

  ngOnInit() {
    this.route.params.subscribe((params) => {
      this.bloqueId = +params['id'];
      this.tituloBloque = this.TEMAS_BLOQUES[this.bloqueId] || 'Bloque';
      this.limpiarEstado();
    });

    const revision = sessionStorage.getItem('intento_revision');
    if (revision) {
      const data = JSON.parse(revision);
      let circuito = data.circuito;

      if (circuito?.circuito) circuito = circuito.circuito;

      this.circuitoGenerado = circuito;
      this.preguntasEjemplo = data.preguntas || [];
      this.resultadoVisible = true;
      sessionStorage.removeItem('intento_revision');
    }
  }

  limpiarEstado() {
    this.circuitoGenerado = null;
    this.preguntasEjemplo = [];
    this.resultadoVisible = false;
  }

  generarEjercicio() {
    this.cargando = true;
    this.errorCircuito = '';

    const payload: any = { bloque: this.bloqueId };

    if (this.bloqueId <= 8) {
      payload.rows = this.rows;
      payload.cols = this.cols;
    }

    if (this.bloqueId > 8) {
      payload.num_sections = this.secciones;
    }

    this.circuitosService.generarCircuito(payload).subscribe({
      next: (res) => {
        if (this.bloqueId > 8) {
          for (let i = 0; i < res.circuito.sections?.length; i++) {
            if (
              i == res.circuito.sections.length - 1 &&
              res.circuito.sections[i].type == 'serie'
            ) {
              this.errorCircuito = 'El circuito no debe contener componentes en serie';
              this.cargando = false;
              this.generarEjercicio(); 
              return;
            }
          }
        }

        this.cargando = false;

        if (res.circuito.cols > 6 || res.circuito.rows > 6) {
          this.errorCircuito = 'Circuito demasiado grande';
          return;
        }

        if (res.circuito.cols < 2 || res.circuito.rows < 2) {
          this.errorCircuito = 'Circuito demasiado pequeño';
          return;
        }

        if (!res.success) {
          this.errorCircuito = 'Error generando circuito';
          return;
        }

        this.circuitoGenerado = res.circuito;
        this.tipoCircuito = res.tipo;
        this.prepararPreguntas(res.circuito);
      },
      error: () => {
        this.cargando = false;
        this.errorCircuito = 'Error backend';
      },
    });
  }

  prepararPreguntas(circuito: any) {
    this.preguntasEjemplo = [];

    if (this.bloqueId === 4) {
      this.generarPreguntasBloque4(circuito);
    } else if (this.bloqueId >= 2 && this.bloqueId <= 7) {
      this.generarBasicas(circuito);
    } else if (this.bloqueId === 8) {
      this.generarPotenciasMonofasico(circuito);
    } else if (this.bloqueId === 9) {
      this.generarTrifasico(circuito);
    } else if (this.bloqueId === 10) {
      this.generarPotenciasTrifasico(circuito);
    }
  }

  obtenerComponentesValidos(comps: any[]) {
    return comps.filter((c) => {
      return (
        Math.abs(c.current) > 1e-6 &&
        Math.abs(c.v_drop) > 1e-6 &&
        c.type !== 'opencircuit'
      );
    });
  }

  generarPreguntasBloque4(circuito: any) {
    // Escenario A: Si Django ya envía el set analítico de preguntas resuelto
    if (circuito.preguntas && circuito.preguntas.length > 0) {
      this.preguntasEjemplo = circuito.preguntas.map((p: any, i: number) => ({
        id: `p${i}`,
        label: p.enunciado,
        valorReal: p.esperado_numerico !== undefined ? p.esperado_numerico : parseFloat(p.esperado),
        unidad: p.unidad || 'V',
        respuestaUsuario: null,
        acertada: undefined
      }));
      return;
    }

    // Escenario B: Fallback predictivo basado en nudos topológicos si vienen vacías
    const nodos = circuito.nodos || [];
    const nodoA = nodos.find((n: any) => n.id === 'N10');
    const nodoB = nodos.find((n: any) => n.id === 'N12');

    const preguntas: any[] = [
      {
        label: 'Tensión de nudo esencial A (Potencial en N10 respecto a GND)',
        valorReal: nodoA?.v_potencial !== undefined ? Math.abs(nodoA.v_potencial) : 15.0,
        unidad: 'V',
      },
      {
        label: 'Tensión de nudo esencial B (Potencial en N12 respecto a GND)',
        valorReal: nodoB?.v_potencial !== undefined ? Math.abs(nodoB.v_potencial) : 10.0,
        unidad: 'V',
      },
      {
        label: 'Diferencia de potencial de control entre bornes de estudio (V_AB)',
        valorReal: (nodoA?.v_potencial !== undefined && nodoB?.v_potencial !== undefined) 
          ? Math.abs(nodoA.v_potencial - nodoB.v_potencial) 
          : 5.0,
        unidad: 'V',
      }
    ];

    this.preguntasEjemplo = preguntas.map((p, i) => ({
      id: `p${i}`,
      ...p,
      respuestaUsuario: null,
      acertada: undefined,
    }));
  }

  generarBasicas(circuito: any) {
    const comps = circuito.componentes || [];
    const nodos = circuito.nodos || [];
    const preguntas: any[] = [];

    if (comps.length) {
      const validos = this.obtenerComponentesValidos(comps);

      if (validos.length) {
        const c = validos[(Math.random() * validos.length) | 0];

        preguntas.push({
          label: `Corriente en ${c.id}`,
          valorReal: Math.abs(Number(c.current) || 0),
          unidad: 'A',
        });

        preguntas.push({
          label: `Caída de tensión en ${c.id}`,
          valorReal: Math.abs(Number(c.v_drop) || 0),
          unidad: 'V',
        });

        if (c.current) {
          preguntas.push({
            label: `Resistencia equivalente en ${c.id}`,
            valorReal: Math.abs(Number(c.v_drop) / Number(c.current)),
            unidad: 'Ω',
          });
        }

        preguntas.push({
          label: `Potencia en ${c.id}`,
          valorReal: Math.abs((Number(c.current) || 0) * (Number(c.v_drop) || 0)),
          unidad: 'W',
        });
      }

      preguntas.push({
        label: 'Número de nodos del circuito',
        valorReal: nodos.length,
        unidad: '',
      });

      preguntas.push({
        label: 'Número de componentes',
        valorReal: comps.length,
        unidad: '',
      });
    }

    this.preguntasEjemplo = preguntas
      .sort(() => Math.random() - 0.5)
      .slice(0, 4)
      .map((p, i) => ({
        id: `p${i}`,
        ...p,
        respuestaUsuario: null,
        acertada: undefined,
      }));
  }

  generarPotenciasMonofasico(circuito: any) {
    const comps = circuito.componentes || [];
    let P_total = 0;
    let Q_total = 0;

    comps.forEach((c: any) => {
      const p_inst = Math.abs((c.current || 0) * (c.v_drop || 0));
      P_total += p_inst;

      if (c.type === 'capacitor' || c.type === 'inductor') {
        Q_total += p_inst;
      }
    });

    const S_total = Math.sqrt(Math.pow(P_total, 2) + Math.pow(Q_total, 2));
    const FP = S_total > 0 ? P_total / S_total : 1;

    const preguntas: any[] = [];

    preguntas.push({
      label: 'Potencia activa total del circuito',
      valorReal: P_total,
      unidad: 'W',
    });
    preguntas.push({
      label: 'Potencia aparente total',
      valorReal: S_total,
      unidad: 'VA',
    });
    preguntas.push({
      label: 'Factor de potencia (cos φ)',
      valorReal: Number(FP.toFixed(2)),
      unidad: '',
    });

    if (comps.length > 0) {
      const c = comps[Math.floor(Math.random() * comps.length)];
      preguntas.push({
        label: `Potencia disipada en el componente ${c.type} de ${c.value}`,
        valorReal: Math.abs((c.current || 0) * (c.v_drop || 0)),
        unidad: 'W',
      });
    }

    this.preguntasEjemplo = preguntas
      .sort(() => Math.random() - 0.5)
      .slice(0, 4)
      .map((p, i) => ({
        id: `p${i}`,
        ...p,
        respuestaUsuario: null,
        acertada: undefined,
      }));
  }

  modulo(val: any): number {
    if (!val) return 0;
    if (typeof val === 'number') return Math.abs(val);
    if (val.re !== undefined && val.im !== undefined) {
      return Math.sqrt(val.re * val.re + val.im * val.im);
    }
    return 0;
  }

  comprobarRespuestas() {
    if (this.preguntasEjemplo.some((p) => p.respuestaUsuario == null)) {
      Swal.fire('Atención', 'Completa todas las preguntas', 'warning');
      return;
    }

    Swal.fire({
      title: '¿Enviar respuestas?',
      icon: 'question',
      showCancelButton: true,
    }).then((result) => {
      if (!result.isConfirmed) return;

      const TOLERANCIA = 0.05;
      let aciertos = 0;

      this.preguntasEjemplo.forEach((p) => {
        const real = Number(p.valorReal);
        const user = Number(p.respuestaUsuario);

        const error = real === 0 ? Math.abs(user) : Math.abs(real - user) / Math.abs(real);
        const ok = error <= TOLERANCIA;

        p.acertada = ok;
        if (ok) aciertos++;
      });

      const fallos = this.preguntasEjemplo.length - aciertos;
      const token = localStorage.getItem('token');
      const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);

      this.http
        .post(
          'http://localhost:8000/api/auth/historial/',
          {
            bloque_id: this.bloqueId,
            aciertos,
            fallos,
            detalle_ejercicio: {
              circuito: this.circuitoGenerado,
              preguntas: this.preguntasEjemplo,
            },
          },
          { headers },
        )
        .subscribe(() => {
          this.resultadoVisible = true;
          this.mensajeResultado = `${aciertos} de ${this.preguntasEjemplo.length} acertadas`;
          Swal.fire('OK', this.mensajeResultado, 'success');
        });
    });
  }

  generarTrifasico(circuito: any) {
    const r = circuito.results || {};
    const params = circuito.params || {};
    const A = r.A || {};
    const B = r.B || {};
    const C = r.C || {};

    const preguntas: any[] = [];

    preguntas.push({
      label: 'Corriente de línea fase A',
      valorReal: this.modulo(A.I_line),
      unidad: 'A',
    });
    preguntas.push({
      label: 'Corriente de línea fase B',
      valorReal: this.modulo(B.I_line),
      unidad: 'A',
    });
    preguntas.push({
      label: 'Corriente de línea fase C',
      valorReal: this.modulo(C.I_line),
      unidad: 'A',
    });
    preguntas.push({
      label: 'Tensión de fase (A)',
      valorReal: this.modulo(A.V_phase),
      unidad: 'V',
    });
    preguntas.push({
      label: 'Tensión de línea',
      valorReal: params.v_line,
      unidad: 'V',
    });
    preguntas.push({
      label: 'Relación Vlínea / Vfase',
      valorReal: params.v_line / this.modulo(A.V_phase || 1),
      unidad: '',
    });

    this.preguntasEjemplo = preguntas
      .sort(() => Math.random() - 0.5)
      .slice(0, 4)
      .map((p, i) => ({
        id: `p${i}`,
        ...p,
        respuestaUsuario: null,
        acertada: undefined,
      }));
  }

  generarPotenciasTrifasico(circuito: any) {
    const r = circuito.results || {};
    const p = circuito.params || {};

    const P = Math.abs(p.P_total || 0);
    const Q = Math.abs(p.Q_total || 0);
    const S = Math.abs(p.S_total || 0);
    const FP = S !== 0 ? P / S : 0;

    const poolPreguntas = [
      { label: 'Potencia activa total del sistema', valorReal: P, unidad: 'W' },
      { label: 'Potencia reactiva total del sistema', valorReal: Q, unidad: 'VAR' },
      { label: 'Potencia aparente total del sistema', valorReal: S, unidad: 'VA' },
      { label: 'Factor de potencia del sistema', valorReal: Number(FP.toFixed(2)), unidad: '' },
      { label: 'Potencia activa fase A', valorReal: Math.abs(r.A?.P || 0), unidad: 'W' },
      { label: 'Potencia activa fase B', valorReal: Math.abs(r.B?.P || 0), unidad: 'W' },
      { label: 'Potencia activa fase C', valorReal: Math.abs(r.C?.P || 0), unidad: 'W' },
    ];

    this.preguntasEjemplo = poolPreguntas
      .sort(() => Math.random() - 0.5)
      .slice(0, 4)
      .map((p, i) => ({
        id: `p${i}`,
        ...p,
        valorMostrado: Number(Number(p.valorReal).toFixed(2)),
        respuestaUsuario: null,
        acertada: undefined,
      }));
  }
}