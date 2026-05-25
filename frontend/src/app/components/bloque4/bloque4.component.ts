import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CircuitosService } from '../../services/circuitos.service';
import { Circuit4ViewComponent } from '../circuit4-view/circuit4-view.component';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import Swal from 'sweetalert2';

@Component({
  selector: 'app-bloque4',
  standalone: true,
  imports: [CommonModule, FormsModule, Circuit4ViewComponent],
  templateUrl: './bloque4.component.html',
  styleUrls: ['./bloque4.component.css'],
})
export class Bloque4Component implements OnInit {
  public circuitData: any = null;
  public problemaSeleccionado = 1;
  public cargando = false;

  public preguntasEjemplo: any[] = [];

  constructor(
    private circuitosService: CircuitosService,
    private http: HttpClient,
  ) {}

  ngOnInit(): void {
    this.cargarCircuitoBloque4(this.problemaSeleccionado);
  }

  onCambiarProblema(id: string | number): void {
    this.problemaSeleccionado = Number(id);
    this.cargarCircuitoBloque4(this.problemaSeleccionado);
  }

  cargarCircuitoBloque4(id: number): void {
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

          if (!res?.success) {
            Swal.fire('Error', 'Backend inválido', 'error');
            return;
          }

          this.circuitData = res.circuito;
          console.log('Circuito recibido:', this.circuitData);

          // 🔥 AQUÍ ESTÁ LA CLAVE
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

  comprobarRespuestas() {
    if (!this.preguntasEjemplo.length) return;

    Swal.fire({
      title: '¿Estás seguro?',
      text: 'Se van a comprobar tus respuestas',
      icon: 'question',
      showCancelButton: true,
      confirmButtonText: 'Sí, comprobar',
      cancelButtonText: 'Cancelar',
    }).then((result) => {
      if (!result.isConfirmed) return;

      let aciertos = 0;
      const TOL = 0.05;

      this.preguntasEjemplo.forEach((p) => {
        const error =
          Math.abs(Number(p.valorReal) - Number(p.respuestaUsuario)) /
          Math.abs(Number(p.valorReal));

        const ok = error <= TOL;

        p.acertada = ok;

        if (ok) aciertos++;
      });

      const fallos = this.preguntasEjemplo.length - aciertos;

      const token = localStorage.getItem('token');

      const headers = new HttpHeaders({
        Authorization: `Bearer ${token}`,
      });

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
}
