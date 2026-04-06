import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpHeaders } from '@angular/common/http'; // Añadido HttpHeaders
import { ActivatedRoute, Router } from '@angular/router';

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
  public historial: any[] = []; // Aquí guardaremos todo lo que venga de Django

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
    private route: ActivatedRoute,
    private router: Router, 
  ) {}

  ngOnInit() {
    this.nombreUsuario = sessionStorage.getItem('uvus')?.toUpperCase() || 'USUARIO';
    console.log(this.nombreUsuario)
    this.cargarDatosDesdeServidor();
  }

  cargarDatosDesdeServidor() {
    const token = localStorage.getItem('token');
    const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);

    // Llamada a tu nuevo endpoint de Django
    this.http.get<any[]>('http://localhost:8000/api/auth/historial/', { headers }).subscribe({
      next: (data) => {
        this.historial = data; // Django ya devuelve el historial del usuario logueado
        this.calcularTotalesGlobales();
      },
      error: (err) => {
        console.error('Error al cargar historial:', err);
        if (err.status === 401) {
          this.router.navigate(['/login']); // Redirigir si el token expiró
        }
      }
    });
  }

  calcularTotalesGlobales() {
    // Sumamos todos los aciertos y fallos de todos los intentos del historial para saber las preguntas totales
    this.aciertos = this.historial.reduce((sum, item) => sum + item.aciertos, 0);
    this.fallos = this.historial.reduce((sum, item) => sum + item.fallos, 0);
    this.totalPreguntasS = this.aciertos + this.fallos;
    this.aciertos.toExponential(1)
  }

  obtenerEstadisticas(bloqueId: number) {
    // Filtramos los intentos que pertenecen a este bloque específico
    const intentosBloque = this.historial.filter(h => h.bloque_id === bloqueId);
    
    const bAciertos = intentosBloque.reduce((sum, h) => sum + h.aciertos, 0);
    const bFallos = intentosBloque.reduce((sum, h) => sum + h.fallos, 0);
    const bTotal = bAciertos + bFallos;

    const exito = bTotal > 0 ? Math.round((bAciertos / 20) * 100) : 0;
    const metaAciertos = 20;
    const progreso = Math.min(Math.round((bAciertos / metaAciertos) * 100), 100);

    return { total: bTotal, aciertos: bAciertos, exito, progreso };
  }

  verRevision(intento: any) {
    // IMPORTANTE: Adaptamos los nombres a los que usa tu BloqueComponent (circuito y preguntas)
    const dataRevision = {
      circuito: intento.detalle_ejercicio.circuito,
      preguntas: intento.detalle_ejercicio.preguntas
    };
    
    sessionStorage.setItem('intento_revision', JSON.stringify(dataRevision));
    // En Django el campo es bloque_id (con guion bajo)
    this.router.navigate(['/bloque', intento.bloque_id, "ejercicio"]);
  }
}