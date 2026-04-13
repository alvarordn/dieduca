import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { CircuitosService } from '../../services/circuitos.service';
import { CircuitViewerComponent } from '../circuit-viewer/circuit-viewer.component';
import Swal from 'sweetalert2';
import { HttpClient, HttpHeaders } from '@angular/common/http';

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
    private http: HttpClient,
  ) {}

  ngOnInit() {
    this.route.params.subscribe((params) => {
      this.bloqueId = +params['id'];
      this.limpiarEstado();
    });

    const revision = sessionStorage.getItem('intento_revision');

    if (revision) {
      const data = JSON.parse(revision);
      console.log(data);
      this.circuitoGenerado = data.circuito;
      this.preguntasEjemplo = data.preguntas;
      this.resultadoVisible = true;
      this.mensajeResultado = 'Revisando intento';
      sessionStorage.removeItem('intento_revision');

      setTimeout(() => {
        window.scrollTo({ top: 500, behavior: 'smooth' });
      }, 500);
    }
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
      Swal.fire('Atención', 'Por favor, completa todas las preguntas.', 'warning');
      return;
    }

    Swal.fire({
      title: '¿Enviar respuestas?',
      text: 'Se calculará tu nota con un 5% de margen de error.',
      icon: 'question',
      showCancelButton: true,
      confirmButtonText: 'Sí, corregir',
      cancelButtonText: 'Revisar',
    }).then((result) => {
      if (result.isConfirmed) {
        let aciertos = 0;
        let fallos = 0;
        
        // Definimos una tolerancia del 5% para considerar una respuesta como correcta
        const tolerancia = 0.05; 

        this.preguntasEjemplo.forEach((p) => {
          // Usamos valor absoluto para comparar solo magnitudes
          const real = Math.abs(p.valorReal);
          const usuario = Math.abs(p.respuestaUsuario);

          // Caso A: El valor es prácticamente cero
          if (real < 1e-6) {
            p.acertada = usuario < 0.01; // Si el real es 0, aceptamos algo muy pequeño
          } 
          // Caso B: Cálculo de ERROR RELATIVO (Diferencia / Real)
          else {
            const errorRelativo = Math.abs(real - usuario) / real;
            p.acertada = errorRelativo <= tolerancia;
          }

          p.acertada ? aciertos++ : fallos++;
        });

        // --- Lógica de envío a Django ---
        const token = localStorage.getItem('token');
        const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);

        const datosEnvio = {
          bloque_id: Number(this.bloqueId),
          aciertos: aciertos,
          fallos: fallos,
          detalle_ejercicio: {
            circuito: this.circuitoGenerado.circuito || this.circuitoGenerado,
            preguntas: this.preguntasEjemplo,
          },
        };

        this.http.post('http://localhost:8000/api/auth/historial/', datosEnvio, { headers })
          .subscribe({
            next: () => {
              this.resultadoVisible = true;
              this.mensajeResultado = `Resultado: ${aciertos} aciertos y ${fallos} fallos.`;
              Swal.fire('¡Guardado!', 'Tu progreso se ha registrado.', 'success');
            },
            error: () => Swal.fire('Error', 'No se pudo guardar el intento.', 'error')
          });
      }
    });
  }

  generarEjercicio() {
    this.cargando = true;
    this.errorCircuito = '';
    const payload = { bloque: this.bloqueId, rows: this.rows, cols: this.cols };

    if (this.rows > 6 || this.cols > 6) {
      this.errorCircuito = 'Las filas/columnas tienen que ser menores que 7';
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
        console.log('Respuesta del servidor:', res);
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
