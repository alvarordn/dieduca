import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { ActivatedRoute, Router } from '@angular/router';

@Component({
  selector: 'app-resultados',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './resultados.component.html',
  styleUrl: './resultados.component.css',
})
export class ResultadosComponent implements OnInit {

  // Variables globales de estadísticas del usuario
  public fallos: number = 0;
  public aciertos: number = 0;
  public totalPreguntasS: number = 0;
  public nombreUsuario: string = '';

  // Aquí guardamos todo el historial que viene desde Django
  public historial: any[] = [];

  // Lista de bloques del temario
  // (esto luego se usa para calcular progreso por tema)
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
    private http: HttpClient,      // para llamadas a la API
    private route: ActivatedRoute,  // por si necesitamos parámetros de ruta
    private router: Router,         // navegación entre páginas
  ) {}

  ngOnInit() {

    // Cogemos el usuario de sessionStorage y lo mostramos en mayúsculas
    this.nombreUsuario =
      sessionStorage.getItem('uvus')?.toUpperCase() || 'USUARIO';

    console.log(this.nombreUsuario);

    // Al iniciar el componente cargamos los datos del backend
    this.cargarDatosDesdeServidor();
  }

  cargarDatosDesdeServidor() {

    // Sacamos el token guardado en localStorage
    const token = localStorage.getItem('token');

    // Creamos headers con autorización Bearer para Django
    const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);

    // Petición al endpoint del historial del usuario
    this.http
      .get<any[]>('http://localhost:8000/api/auth/historial/', { headers })
      .subscribe({

        // Si todo va bien
        next: (data) => {
          console.log('Datos recibidos de Django:', data);

          // Guardamos el historial recibido
          this.historial = data;

          // Calculamos estadísticas globales
          this.calcularTotalesGlobales();
        },

        // Si hay error en la API
        error: (err) => {
          console.error('Error al cargar historial:', err);

          // Si el token no es válido, mandamos al login
          if (err.status === 401) {
            this.router.navigate(['/login']);
          }
        },
      });
  }

  calcularTotalesGlobales() {

    // 🔴 OJO: esto está un poco raro, porque se recalcula luego igual
    // (pero lo dejo tal cual tu lógica)

    if (this.aciertos > 20){
      this.aciertos = 20;
    }

    // Sumamos todos los aciertos del historial
    this.aciertos = this.historial.reduce(
      (sum, item) => sum + item.aciertos,
      0,
    );

    // Sumamos todos los fallos del historial
    this.fallos = this.historial.reduce(
      (sum, item) => sum + item.fallos,
      0,
    );

    // Total de preguntas realizadas
    this.totalPreguntasS = this.aciertos + this.fallos;

    // (esto no hace nada porque no se guarda el resultado)
    this.aciertos.toExponential(1);
  }

  obtenerEstadisticas(bloqueId: number) {

    // Filtramos solo los intentos del bloque actual
    const intentosBloque = this.historial.filter(
      (h) => h.bloque_id === bloqueId,
    );

    // Sumamos aciertos del bloque
    const bAciertos = intentosBloque.reduce(
      (sum, h) => sum + h.aciertos,
      0,
    );

    // Sumamos fallos del bloque
    const bFallos = intentosBloque.reduce(
      (sum, h) => sum + h.fallos,
      0,
    );

    // Total del bloque
    const bTotal = bAciertos + bFallos;

    // Porcentaje de éxito (basado en 20 preguntas objetivo)
    const exito = bTotal > 0 ? Math.round((bAciertos / 20) * 100) : 0;

    // Progreso máximo limitado a 100%
    const metaAciertos = 20;
    const progreso = Math.min(
      Math.round((bAciertos / metaAciertos) * 100),
      100,
    );

    // Devolvemos stats del bloque
    return { total: bTotal, aciertos: bAciertos, exito, progreso };
  }

  verRevision(intento: any) {

    // Debug: ver lo que devuelve Django
    console.log('Datos brutos de Django:', intento.detalle_ejercicio);

    // Cogemos el circuito del intento
    let circuito = intento.detalle_ejercicio?.circuito;

    // Si viene doble anidado, lo corregimos
    if (circuito?.circuito) {
      circuito = circuito.circuito;
    }

    // Preparamos datos para la pantalla de revisión
    const dataRevision = {
      circuito: circuito,
      preguntas: intento.detalle_ejercicio?.preguntas || [],
    };

    // Guardamos en sessionStorage para usarlo en otra vista
    sessionStorage.setItem(
      'intento_revision',
      JSON.stringify(dataRevision),
    );

    // Navegamos al ejercicio del bloque
    this.router.navigate([
      '/bloque',
      intento.bloque_id,
      'ejercicio',
    ]);
  }
}