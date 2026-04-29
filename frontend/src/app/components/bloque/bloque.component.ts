import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { CircuitosService } from '../../services/circuitos.service';
import { CircuitViewerComponent } from '../circuit-viewer/circuit-viewer.component';
import { ThreePhaseViewerComponent } from '../three-phase-viewer/three-phase-viewer.component';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import Swal from 'sweetalert2';

@Component({
  selector: 'app-bloque',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    CircuitViewerComponent,
    ThreePhaseViewerComponent,
  ],
  templateUrl: './bloque.component.html',
  styleUrl: './bloque.component.css',
})
export class BloqueComponent implements OnInit {

  // id del bloque que viene por la URL (nivel del ejercicio)
  public bloqueId = 0;

  // estado de carga cuando se genera el circuito
  public cargando = false;

  // mensaje de error si algo falla al generar el circuito
  public errorCircuito = '';

  // aquí guardo el circuito que me devuelve el backend
  public circuitoGenerado: any = null;

  // mensaje final con el resultado del intento
  public mensajeResultado = '';

  // controla si se muestran los resultados o no
  public resultadoVisible = false;

  // tipo de circuito (mono o trifásico)
  tipoCircuito: 'monofasico' | 'trifasico' | null = null;

  // lista de preguntas del ejercicio
  preguntasEjemplo: any[] = [];

  // tamaño del circuito en pantalla (grid)
  rows = 2;
  cols = 3;

  // número de secciones (solo para trifásico)
  secciones: number = 2;

  constructor(
    private route: ActivatedRoute,
    private circuitosService: CircuitosService,
    private http: HttpClient,
  ) {}

  ngOnInit() {
    // saco el id del bloque desde la URL
    this.route.params.subscribe((params) => {
      this.bloqueId = +params['id'];
      this.limpiarEstado();
    });

    // si vienes de revisar un intento anterior, lo cargo aquí
    const revision = sessionStorage.getItem('intento_revision');

    if (revision) {
      const data = JSON.parse(revision);

      // a veces el circuito viene anidado raro, por eso lo arreglo aquí
      let circuito = data.circuito;
      if (circuito?.circuito) circuito = circuito.circuito;

      this.circuitoGenerado = circuito;
      this.preguntasEjemplo = data.preguntas || [];
      this.resultadoVisible = true;

      sessionStorage.removeItem('intento_revision');
    }
  }

  // resetea todo cuando cambias de bloque
  limpiarEstado() {
    this.circuitoGenerado = null;
    this.preguntasEjemplo = [];
    this.resultadoVisible = false;
  }

  // llama al backend para generar el circuito
  generarEjercicio() {
    this.cargando = true;
    this.errorCircuito = '';

    const payload: any = {
      bloque: this.bloqueId,
    };

    // si es monofásico, mando filas y columnas
    if (this.bloqueId <= 8) {
      payload.rows = this.rows;
      payload.cols = this.cols;
    }

    // si es trifásico, mando número de secciones
    if (this.bloqueId > 8) {
      payload.num_sections = this.secciones;
    }

    this.circuitosService.generarCircuito(payload).subscribe({
      next: (res) => {
        this.cargando = false;

        console.log('Respuesta circuito: ', res);

        // validaciones básicas de tamaño del circuito
        if (res.circuito.cols > 6 || res.circuito.rows > 6) {
          this.errorCircuito = 'Circuito demasiado grande';
          return;
        }

        if (res.circuito.cols < 2 || res.circuito.rows < 2) {
          this.errorCircuito = 'Circuito demasiado pequeño';
          return;
        }

        // validaciones extra para trifásico
        if (res.tipo === 'trifasico') {
          if (res.circuito.sections.length > 6) {
            this.errorCircuito = 'Demasiadas secciones';
            return;
          }
          if (res.circuito.sections.length < 2) {
            this.errorCircuito = 'Muy pocas secciones';
            return;
          }
        }

        if (!res.success) {
          this.errorCircuito = 'Error generando circuito';
          return;
        }

        // guardo circuito y tipo
        this.circuitoGenerado = res.circuito;
        this.tipoCircuito = res.tipo;

        // genero preguntas según el bloque
        this.prepararPreguntas(res.circuito);
      },

      error: () => {
        this.cargando = false;
        this.errorCircuito = 'Error backend';
      },
    });
  }

  // según el bloque, genero preguntas distintas
  prepararPreguntas(circuito: any) {
    this.preguntasEjemplo = [];

    if (this.bloqueId >= 2 && this.bloqueId <= 7) {
      this.generarBasicas(circuito);
      return;
    }

    if (this.bloqueId === 8) {
      this.generarPotenciasMonofasico(circuito);
      return;
    }

    if (this.bloqueId === 9) {
      this.generarTrifasico(circuito);
      return;
    }

    if (this.bloqueId === 10) {
      this.generarPotenciasTrifasico(circuito);
      return;
    }
  }

  // preguntas básicas de monofásico
  generarBasicas(circuito: any) {
    const comps = circuito.componentes || [];

    const preguntasDisponibles: any[] = [];

    // corriente de un componente random
    if (comps.length) {
      const c = comps[(Math.random() * comps.length) | 0];

      preguntasDisponibles.push({
        label: `Corriente en ${c.id}`,
        valorReal: c.current,
        unidad: 'A',
      });
    }

    // caída de tensión
    if (comps.length) {
      const c = comps[(Math.random() * comps.length) | 0];

      preguntasDisponibles.push({
        label: `Caída de tensión en ${c.id}`,
        valorReal: c.v_drop,
        unidad: 'V',
      });
    }

    // resistencia (ley de Ohm)
    if (comps.length) {
      const c = comps[(Math.random() * comps.length) | 0];

      if (c.current) {
        preguntasDisponibles.push({
          label: `Resistencia en ${c.id}`,
          valorReal: c.v_drop / c.current,
          unidad: 'Ω',
        });
      }
    }

    // componente con más corriente
    if (comps.length) {
      const mayor = comps.reduce((a: any, b: any) =>
        Math.abs(a.current) > Math.abs(b.current) ? a : b,
      );

      preguntasDisponibles.push({
        label: `Corriente máxima del circuito`,
        valorReal: Math.abs(mayor.current),
        unidad: 'A',
      });
    }

    // mayor caída de tensión
    if (comps.length) {
      const mayor = comps.reduce((a: any, b: any) =>
        Math.abs(a.v_drop) > Math.abs(b.v_drop) ? a : b,
      );

      preguntasDisponibles.push({
        label: `Mayor caída de tensión`,
        valorReal: Math.abs(mayor.v_drop),
        unidad: 'V',
      });
    }

    // me quedo con 4 aleatorias para que no sea siempre igual
    const seleccionadas = preguntasDisponibles
      .sort(() => Math.random() - 0.5)
      .slice(0, 4);

    this.preguntasEjemplo = seleccionadas.map((p, i) => ({
      id: `p${i}`,
      ...p,
      respuestaUsuario: null,
      acertada: undefined,
    }));
  }

  // bloque 8: potencias monofásicas
  generarPotenciasMonofasico(circuito: any) {
    const comps = circuito.componentes || [];

    let P_total = 0;

    // sumo potencia total del circuito
    comps.forEach((c: any) => {
      P_total += Math.abs((c.current || 0) * (c.v_drop || 0));
    });

    this.preguntasEjemplo = [];

    // potencia total
    this.preguntasEjemplo.push({
      id: 'P',
      label: 'Potencia total del circuito',
      valorReal: P_total,
      respuestaUsuario: null,
      unidad: 'W',
      acertada: undefined,
    });

    // potencia de un componente random
    if (comps.length) {
      const c = comps[(Math.random() * comps.length) | 0];

      this.preguntasEjemplo.push({
        id: 'Pc',
        label: `Potencia en ${c.id}`,
        valorReal: Math.abs((c.current || 0) * (c.v_drop || 0)),
        respuestaUsuario: null,
        unidad: 'W',
        acertada: undefined,
      });
    }

    // componente con más potencia
    if (comps.length) {
      const max = comps.reduce((a: any, b: any) => {
        const Pa = Math.abs((a.current || 0) * (a.v_drop || 0));
        const Pb = Math.abs((b.current || 0) * (b.v_drop || 0));
        return Pa > Pb ? a : b;
      });

      this.preguntasEjemplo.push({
        id: 'Pmax',
        label: 'Mayor potencia',
        valorReal: Math.abs((max.current || 0) * (max.v_drop || 0)),
        respuestaUsuario: null,
        unidad: 'W',
        acertada: undefined,
      });
    }

    // check de suma
    this.preguntasEjemplo.push({
      id: 'Pcheck',
      label: 'Suma total de potencias',
      valorReal: P_total,
      respuestaUsuario: null,
      unidad: 'W',
      acertada: undefined,
    });
  }

  // trifásico básico
  generarTrifasico(circuito: any) {
    const r = circuito.results || {};
    const A = r.A || {};
    const B = r.B || {};
    const C = r.C || {};

    this.preguntasEjemplo = [];

    // corrientes y tensiones por fase
    this.preguntasEjemplo.push({
      id: 'IA',
      label: 'Corriente fase A',
      valorReal: this.modulo(A.I_line),
      unidad: 'A',
      respuestaUsuario: null,
      acertada: undefined,
    });

    this.preguntasEjemplo.push({
      id: 'IB',
      label: 'Corriente fase B',
      valorReal: this.modulo(B.I_line),
      unidad: 'A',
      respuestaUsuario: null,
      acertada: undefined,
    });

    this.preguntasEjemplo.push({
      id: 'VA',
      label: 'Tensión fase A',
      valorReal: this.modulo(A.V_phase),
      unidad: 'V',
      respuestaUsuario: null,
      acertada: undefined,
    });

    this.preguntasEjemplo.push({
      id: 'VC',
      label: 'Tensión fase C',
      valorReal: this.modulo(C.V_phase),
      unidad: 'V',
      respuestaUsuario: null,
      acertada: undefined,
    });

    // fase con más corriente
    const fases = [
      { name: 'A', val: this.modulo(A.I_line) },
      { name: 'B', val: this.modulo(B.I_line) },
      { name: 'C', val: this.modulo(C.I_line) },
    ];

    const max = fases.reduce((a, b) => (a.val > b.val ? a : b));

    this.preguntasEjemplo.push({
      id: 'IMAX',
      label: 'Fase con mayor corriente',
      valorReal: max.name,
      respuestaUsuario: null,
      unidad: '',
      acertada: undefined,
    });

    // diferencia de tensiones
    this.preguntasEjemplo.push({
      id: 'Veq',
      label: 'Diferencia de tensión A-B',
      valorReal: Math.abs(this.modulo(A.V_phase) - this.modulo(B.V_phase)),
      unidad: 'V',
      respuestaUsuario: null,
      acertada: undefined,
    });

    // selecciono 4 aleatorias
    const seleccionadas = this.preguntasEjemplo
      .sort(() => Math.random() - 0.5)
      .slice(0, 4);

    this.preguntasEjemplo = seleccionadas.map((p, i) => ({
      id: `p${i}`,
      ...p,
      respuestaUsuario: null,
      acertada: undefined,
    }));
  }

  // potencias trifásicas
  generarPotenciasTrifasico(circuito: any) {
    const r = circuito.results || {};

    let P = 0,
      Q = 0,
      S = 0;

    // sumo potencias de todas las fases
    Object.values(r).forEach((f: any) => {
      P += Math.abs(f.P || 0);
      Q += Math.abs(f.Q || 0);
      S += Math.abs(f.S || 0);
    });

    this.preguntasEjemplo.push(
      {
        id: 'P',
        label: 'Potencia activa total',
        valorReal: P,
        unidad: 'W',
        respuestaUsuario: null,
        acertada: undefined,
      },
      {
        id: 'Q',
        label: 'Potencia reactiva total',
        valorReal: Q,
        unidad: 'VAR',
        respuestaUsuario: null,
        acertada: undefined,
      },
      {
        id: 'S',
        label: 'Potencia aparente total',
        valorReal: S,
        unidad: 'VA',
        respuestaUsuario: null,
        acertada: undefined,
      },
    );
  }

  // calcula módulo (para números complejos en trifásico)
  modulo(val: any): number {
    if (!val) return 0;
    if (typeof val === 'number') return Math.abs(val);

    const s = val.toString();
    const parts = s.match(/[+-]?\d+(\.\d+)?/g) || [];

    const r = Number(parts[0] || 0);
    const i = Number(parts[1] || 0);

    return Math.sqrt(r * r + i * i);
  }

  // comprobar respuestas del alumno
  comprobarRespuestas() {
    if (this.preguntasEjemplo.some((p) => p.respuestaUsuario == null)) {
      Swal.fire('Atención', 'Completa todas las preguntas', 'warning');
      return;
    }

    Swal.fire({
      title: '¿Enviar respuestas?',
      icon: 'question',
      showCancelButton: true,
      confirmButtonText: 'Sí, enviar',
    }).then((result) => {
      if (!result.isConfirmed) return;

      let aciertos = 0;

      // comparo respuestas con un margen de error del 5%
      this.preguntasEjemplo.forEach((p) => {
        const ok =
          Math.abs(p.valorReal - p.respuestaUsuario) / (p.valorReal || 1) <= 0.05;

        p.acertada = ok;

        if (ok) aciertos++;
      });

      const fallos = this.preguntasEjemplo.length - aciertos;

      const token = localStorage.getItem('token');
      const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);

      // guardo resultado en backend (historial)
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
}