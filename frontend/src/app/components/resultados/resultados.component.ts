import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';

@Component({
  selector: 'app-resultados',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './resultados.component.html',
  styleUrl: './resultados.component.css',
})
export class ResultadosComponent implements OnInit {

  public fallos: number = 0;
  public aciertos: number = 0;
  public totalPreguntasS: number = 0;
  public nombreUsuario: string = '';

  public historial: any[] = [];

  public bloques = [
    { id: 1, nombre: 'Conceptos fundamentales y leyes de Kirchhoff' },
    { id: 2, nombre: 'Circuitos Resistivos con Generadores Ideales' },
    { id: 3, nombre: 'Fuentes reales y circuitos equivalentes' },
    { id: 4, nombre: 'Técnicas de análisis de circuitos' },
    { id: 5, nombre: 'Componentes dinámicos' },
    { id: 6, nombre: 'Análisis de circuitos de CC en distintos regímenes' },
    { id: 7, nombre: 'Resolución de circuitos de CA sinusoidal' },
    { id: 8, nombre: 'Potencia y energía en CA sinusoidal' },
    { id: 9, nombre: 'Circuitos trifásicos' },
    { id: 10, nombre: 'Potencia en circuitos trifásicos equilibrados' },
    { id: 11, nombre: 'Fundamentos de máquinas eléctricas' },
  ];

  constructor(
    private http: HttpClient,
    private router: Router,
  ) {}

  ngOnInit() {

    this.nombreUsuario =
      sessionStorage.getItem('uvus')?.toUpperCase() || 'USUARIO';

    this.cargarDatosDesdeServidor();
  }

  cargarDatosDesdeServidor() {

    const token = localStorage.getItem('token');

    const headers = new HttpHeaders({
      Authorization: `Bearer ${token}`,
    });

    this.http.get<any[]>(
      'http://localhost:8000/api/auth/historial/',
      { headers }
    ).subscribe({

      next: (data) => {
        this.historial = data;
        this.calcularTotalesGlobales();
      },

      error: (err) => {
        console.error('Error historial:', err);

        if (err.status === 401) {
          this.router.navigate(['/login']);
        }
      }

    });
  }

  calcularTotalesGlobales() {

    this.aciertos = this.historial.reduce(
      (sum, item) => sum + item.aciertos,
      0,
    );

    this.fallos = this.historial.reduce(
      (sum, item) => sum + item.fallos,
      0,
    );

    this.totalPreguntasS = this.aciertos + this.fallos;
  }

  obtenerEstadisticas(bloqueId: number) {

    const intentos = this.historial.filter(
      h => h.bloque_id === bloqueId
    );

    const aciertos = intentos.reduce(
      (sum, h) => sum + h.aciertos,
      0,
    );

    const fallos = intentos.reduce(
      (sum, h) => sum + h.fallos,
      0,
    );

    const total = aciertos + fallos;

    const progreso = total > 0
      ? Math.min(Math.round((aciertos / 20) * 100), 100)
      : 0;

    const exito = total > 0
      ? Math.round((aciertos / 20) * 100)
      : 0;

    return {
      aciertos,
      fallos,
      total,
      progreso,
      exito
    };
  }

  verRevision(intento: any) {

    console.log('Intento seleccionado:', intento);

    let circuito = intento.detalle_ejercicio?.circuito;

    // FIX: evitar doble anidación
    if (circuito?.circuito) {
      circuito = circuito.circuito;
    }

    const dataRevision = {
      circuito: circuito,
      preguntas: intento.detalle_ejercicio?.preguntas || [],
    };

    // ❗ IMPORTANTE: limpiar antes de guardar nuevo
    sessionStorage.removeItem('intento_revision');

    sessionStorage.setItem(
      'intento_revision',
      JSON.stringify(dataRevision),
    );

    // navegación normal
    this.router.navigate([
      '/bloque',
      intento.bloque_id,
      'ejercicio',
    ]);
  }
}