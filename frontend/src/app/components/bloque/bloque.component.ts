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
  // El ID que pillo de la URL para saber en qué bloque estamos
  public bloqueId = 0;

  // Los nombres de los temas para ponerlos en el título
  TEMAS_BLOQUES: { [key: number]: string } = {
    1: 'Conceptos fundamentales y leyes de Kirchhof',
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

  // Para enseñar el spinner de carga o los fallos
  public cargando = false;
  public errorCircuito = '';

  // Aquí guardo el objeto que me escupe el Python
  public circuitoGenerado: any = null;

  // Para el texto final de si has aprobado o no
  public mensajeResultado = '';
  public resultadoVisible = false;

  // Si es de una fase o de tres
  tipoCircuito: 'monofasico' | 'trifasico' | null = null;

  // El array con las preguntas que le salen al usuario
  preguntasEjemplo: any[] = [];

  // Tamaño de la rejilla por defecto
  rows = 2;
  cols = 3;

  public tituloBloque = '';

  // Cuántas partes tiene el trifásico
  secciones: number = 2;

  constructor(
    private route: ActivatedRoute,
    private circuitosService: CircuitosService,
    private http: HttpClient,
  ) {}

  // Nada más entrar, miro qué bloque es y si hay algo guardado de antes
  ngOnInit() {
    // Me suscribo a los parámetros para pillar el ID
    this.route.params.subscribe((params) => {
      this.bloqueId = +params['id'];

      // Pillo el nombre del mapa de arriba
      this.tituloBloque = this.TEMAS_BLOQUES[this.bloqueId] || 'Bloque';

      // Si cambio de bloque, que no se quede lo viejo pintado
      this.limpiarEstado();
    });

    // Por si el usuario ha dado a "revisar" desde el historial
    const revision = sessionStorage.getItem('intento_revision');

    if (revision) {
      const data = JSON.parse(revision);
      let circuito = data.circuito;

      // Un poco de limpieza por si los datos vienen raros
      if (circuito?.circuito) circuito = circuito.circuito;

      this.circuitoGenerado = circuito;
      this.preguntasEjemplo = data.preguntas || [];
      this.resultadoVisible = true;

      // Lo borro para que no salga siempre al recargar
      sessionStorage.removeItem('intento_revision');
    }
  }

  // Para dejarlo todo a cero
  limpiarEstado() {
    this.circuitoGenerado = null;
    this.preguntasEjemplo = [];
    this.resultadoVisible = false;
  }

  // Función gorda para pedir el circuito al server
  generarEjercicio() {
    this.cargando = true;
    this.errorCircuito = '';

    const payload: any = { bloque: this.bloqueId };

    // Si es un circuito normal, le paso filas y columnas
    if (this.bloqueId <= 8) {
      payload.rows = this.rows;
      payload.cols = this.cols;
    }

    // Si es de los últimos, le paso las secciones
    if (this.bloqueId > 8) {
      payload.num_sections = this.secciones;
    }

    // Llamada al servicio
    this.circuitosService.generarCircuito(payload).subscribe({
      next: (res) => {
        // Validación rara que nos han pedido para los trifásicos
        if (this.bloqueId > 8) {
          for (let i = 0; i < res.circuito.sections.length; i++) {
            if (
              i == res.circuito.sections.length - 1 &&
              res.circuito.sections[i].type == 'serie'
            ) {
              this.errorCircuito =
                'El circuito no debe contener componentes en serie';
              this.cargando = false;
              this.generarEjercicio(); // Reintento automático
              return;
            }
          }
        }

        this.cargando = false;
        console.log('Circuito:', res);

        // Que no se nos rompa el layout si el backend manda una burrada
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

        // Si todo va bien, guardo y saco las preguntas
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

  // Según el bloque en el que estemos, tiro por una función de preguntas u otra
  prepararPreguntas(circuito: any) {
    this.preguntasEjemplo = [];

    console.log('Circuito recibido:', circuito);

    if (this.bloqueId >= 2 && this.bloqueId <= 7) {
      this.generarBasicas(circuito);
    } else if (this.bloqueId === 8) {
      this.generarPotenciasMonofasico(circuito);
    } else if (this.bloqueId === 9) {
      this.generarTrifasico(circuito);
    } else if (this.bloqueId === 10) {
      this.generarPotenciasTrifasico(circuito);
    }
  }

  // Para no preguntar por cables o cosas que no tienen valores
  obtenerComponentesValidos(comps: any[]) {
    return comps.filter((c) => {
      return (
        Math.abs(c.current) > 1e-6 &&
        Math.abs(c.v_drop) > 1e-6 &&
        c.type !== 'opencircuit'
      );
    });
  }

  // Genera preguntas típicas de V, I, R y P
  generarBasicas(circuito: any) {
    const comps = circuito.componentes || [];
    const nodos = circuito.nodos || [];
    const preguntas: any[] = [];

    if (comps.length) {
      const validos = this.obtenerComponentesValidos(comps);

      if (validos.length) {
        // Pillo un componente al azar de los que valen
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
          valorReal: Math.abs(
            (Number(c.current) || 0) * (Number(c.v_drop) || 0),
          ),
          unidad: 'W',
        });
      }

      // Preguntas de relleno sobre el dibujo
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

    // Mezclo un poco y me quedo con 4
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

  // Lógica para potencias en circuitos de una sola fase
  generarPotenciasMonofasico(circuito: any) {
    const comps = circuito.componentes || [];
    let P_total = 0;
    let Q_total = 0;

    comps.forEach((c: any) => {
      // Cálculo básico de potencia
      const p_inst = Math.abs((c.current || 0) * (c.v_drop || 0));
      P_total += p_inst;

      // Si es un condensador o bobina, lo meto en la reactiva
      if (c.type === 'capacitor' || c.type === 'inductor') {
        Q_total += p_inst;
      }
    });

    // Pitágoras para la potencia aparente
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

  // Para sacar el valor de un número complejo (módulo)
  modulo(val: any): number {
    if (!val) return 0;
    if (typeof val === 'number') return Math.abs(val);
    if (val.re !== undefined && val.im !== undefined) {
      return Math.sqrt(val.re * val.re + val.im * val.im);
    }
    return 0;
  }

  // Cuando el usuario le da al botón de enviar
  comprobarRespuestas() {
    // Que no se dejen nada vacío
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

      // Margen de error del 5% por si hay redondeos
      const TOLERANCIA = 0.05;
      let aciertos = 0;

      this.preguntasEjemplo.forEach((p) => {
        const real = Number(p.valorReal);
        const user = Number(p.respuestaUsuario);

        // Si es 0 comparo a pelo, si no, saco el porcentaje de error
        const error =
          real === 0 ? Math.abs(user) : Math.abs(real - user) / Math.abs(real);
        const ok = error <= TOLERANCIA;

        p.acertada = ok;
        if (ok) aciertos++;
      });

      const fallos = this.preguntasEjemplo.length - aciertos;
      const token = localStorage.getItem('token');
      const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);

      // Guardo el resultado en el historial de la base de datos
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

  // Preguntas para el sistema trifásico (líneas y fases)
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

  // Preguntas de potencias pero para trifásica (sistema completo)
  generarPotenciasTrifasico(circuito: any) {
    const r = circuito.results || {};
    let P = 0;
    let Q = 0;
    let S = 0;

    // Sumo lo de las tres fases (A, B, C)
    Object.values(r).forEach((f: any) => {
      P += Math.abs(f.P || 0);
      Q += Math.abs(f.Q || 0);
      S += Math.abs(f.S || 0);
    });

    const FP = S !== 0 ? P / S : 0;
    const preguntas: any[] = [];

    preguntas.push({
      label: 'Potencia activa total del sistema',
      valorReal: P,
      unidad: 'W',
    });
    preguntas.push({
      label: 'Potencia reactiva total del sistema',
      valorReal: Q,
      unidad: 'VAR',
    });
    preguntas.push({
      label: 'Potencia aparente total del sistema',
      valorReal: S,
      unidad: 'VA',
    });
    preguntas.push({
      label: 'Factor de potencia del sistema',
      valorReal: FP,
      unidad: '',
    });
    preguntas.push({
      label: 'Potencia activa fase A',
      valorReal: Math.abs(r.A?.P || 0),
      unidad: 'W',
    });
    preguntas.push({
      label: 'Potencia activa fase B',
      valorReal: Math.abs(r.B?.P || 0),
      unidad: 'W',
    });
    preguntas.push({
      label: 'Potencia activa fase C',
      valorReal: Math.abs(r.C?.P || 0),
      unidad: 'W',
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
}
