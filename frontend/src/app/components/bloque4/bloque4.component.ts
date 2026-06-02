// Import básico de Angular (lo típico para que el componente funcione)
import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

// Servicio que habla con el backend para generar circuitos
import { CircuitosService } from '../../services/circuitos.service';

// Componente que pinta el circuito (el SVG rollo visual)
import { Circuit4ViewComponent } from '../circuit4-view/circuit4-view.component';

// Para hacer peticiones HTTP al backend
import { HttpClient, HttpHeaders } from '@angular/common/http';

// Para navegar entre páginas
import { Router } from '@angular/router';

// Popups bonitos tipo alert
import Swal from 'sweetalert2';

@Component({
  selector: 'app-bloque4',
  standalone: true,
  imports: [CommonModule, FormsModule, Circuit4ViewComponent],
  templateUrl: './bloque4.component.html',
  styleUrls: ['./bloque4.component.css'],
})
export class Bloque4Component implements OnInit {
  // circuito que viene del backend
  circuitData: any = null;

  // array con las preguntas del ejercicio
  preguntasEjemplo: any[] = [];

  // problema seleccionado en el dropdown
  problemaSeleccionado = 1;

  // loading mientras carga el backend
  cargando = false;

  // modo revisión (cuando ves un intento ya hecho)
  modoRevision = false;

  constructor(
    private circuitosService: CircuitosService, // pide circuitos
    private http: HttpClient, // manda datos al backend
    private router: Router, // navegación entre páginas
  ) {}

  ngOnInit(): void {
    // mira si venimos de “ver intento anterior”
    const revision = sessionStorage.getItem('intento_revision');

    if (revision) {
      // activamos modo solo lectura
      this.modoRevision = true;

      // sacamos datos guardados en JSON
      const data = JSON.parse(revision);

      // cargamos circuito tal cual estaba en el intento
      this.circuitData = data.circuito;

      // reconstruimos preguntas con sus respuestas ya hechas
      this.preguntasEjemplo = (data.preguntas || []).map(
        (p: any, i: number) => ({
          id: i,
          label: p.label,
          unidad: p.unidad,
          valorReal: p.valorReal,
          respuestaUsuario: p.respuestaUsuario,
          acertada: p.acertada,
        }),
      );

      return; // cortamos aquí porque ya estamos en modo revisión
    }

    // si no hay revisión → cargamos ejercicio normal
    this.cargarCircuito(this.problemaSeleccionado);
  }

  // cuando cambias el select de problemas
  onCambiarProblema(id: string | number) {
    // si estás en modo revisión no dejas tocar nada
    if (this.modoRevision) return;

    this.problemaSeleccionado = Number(id);

    // recarga circuito nuevo
    this.cargarCircuito(this.problemaSeleccionado);
  }

  // pide circuito al backend
  cargarCircuito(id: number) {
    this.cargando = true;

    this.circuitosService
      .generarCircuito({
        bloque: 4,
        rows: 3,
        cols: 3,
        plantilla: id,
      })
      .subscribe({
        next: (res: any) => {
          this.cargando = false;

          // guardamos circuito
          this.circuitData = res.circuito;

          // convertimos preguntas del backend a formato del front
          this.preguntasEjemplo = (res.circuito?.preguntas?.items || []).map(
            (p: any, i: number) => ({
              id: i,
              label: p.label,
              unidad: p.unidad,
              valorReal: p.solucion,
              respuestaUsuario: null,
              acertada: undefined,
            }),
          );
        },

        error: () => {
          this.cargando = false;
          Swal.fire('Error', 'Servidor no disponible', 'error');
        },
      });
  }

  // comprobar respuestas del alumno
  comprobarRespuestas() {
    // ver si hay inputs vacíos
    const incompletas = this.preguntasEjemplo.some(
      (p) =>
        p.respuestaUsuario === null ||
        p.respuestaUsuario === undefined ||
        p.respuestaUsuario === '',
    );

    if (incompletas) {
      Swal.fire({
        icon: 'error',
        title: 'Faltan respuestas',
        text: 'Tienes que completar todas las preguntas antes de enviar',
      });
      return; // no deja seguir
    }

    const TOL = 0.05; // margen de error del 5%

    let aciertos = 0;

    // aquí calculamos resultados pero SIN pintar aún en UI
    const resultados = this.preguntasEjemplo.map((p) => {
      const error =
        Math.abs(p.valorReal - p.respuestaUsuario) / Math.abs(p.valorReal);

      const ok = error <= TOL;

      if (ok) aciertos++;

      return {
        ...p,
        acertada: ok,
      };
    });

    const fallos = this.preguntasEjemplo.length - aciertos;

    const token = localStorage.getItem('token');

    const headers = new HttpHeaders({
      Authorization: `Bearer ${token}`,
    });

    // confirmación antes de enviar
    Swal.fire({
      title: '¿Estás seguro?',
      text: 'Vas a enviar tus respuestas para corregir el ejercicio',
      icon: 'question',
      showCancelButton: true,
      confirmButtonText: 'Sí, enviar',
      cancelButtonText: 'Cancelar',
    }).then((result) => {
      // si cancela → no hace nada
      if (!result.isConfirmed) return;

      // si acepta → ahora sí pintamos resultados en pantalla
      this.preguntasEjemplo = resultados;

      // mandamos al backend el intento
      this.http
        .post(
          'http://localhost:8000/api/auth/historial/',
          {
            bloque_id: 4,
            aciertos,
            fallos,
            detalle_ejercicio: {
              circuito: this.circuitData,
              preguntas: this.preguntasEjemplo,
            },
          },
          { headers },
        )
        .subscribe(() => {
          Swal.fire(
            'Resultado',
            `${aciertos}/${this.preguntasEjemplo.length}`,
            'success',
          );
        });
    });
  }

  borrarRespuestas() {
    this.preguntasEjemplo = this.preguntasEjemplo.map((p) => ({
      ...p,
      respuestaUsuario: null,
      acertada: undefined,
    }));

    Swal.fire({
      icon: 'info',
      title: 'Respuestas borradas',
      timer: 1200,
      showConfirmButton: false,
    });
  }

  // ir a pantalla de resultados
  verRevision() {
    this.router.navigate(['/resultados']);
  }
}
