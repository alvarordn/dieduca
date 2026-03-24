import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, Input } from '@angular/core';
import { ActivatedRoute, Route, RouterLink } from '@angular/router';

@Component({
  selector: 'app-resultados',
  imports: [CommonModule],
  templateUrl: './resultados.component.html',
  styleUrl: './resultados.component.css',
})
export class ResultadosComponent {
  public fallos: string = '';
  public aciertos: string = '';
  public exito = 0;
  public idUsuario = 0;
  public totalPreguntasS: string = '';
  public totalPreguntas = 0;
  public nombreUsuario: string = '';
  // Inicializamos los bloques en un array de objetos
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
    { id: 11, nombre: 'Fundamentos de máquinas eléctricas' }
  ];

  constructor(
    private http: HttpClient,
    private route: ActivatedRoute,
  ) {
    // Inicializamos los aciertos, fallos o totalPreguntas a 0 o lo que haya en el backend/localstorage
    this.aciertos = localStorage.getItem('Aciertos') || '0';
    this.fallos = localStorage.getItem('Fallos') || '0';
    this.totalPreguntasS = localStorage.getItem('TotalPreguntas') || '0';
  }

  ngOnInit() {
    // Obtenemos de la ruta el id de usuario
    this.idUsuario = this.route.snapshot.params['id'];
    console.log(this.idUsuario);

    // Obtenemos el uvus del usuario
    this.nombreUsuario =
      localStorage.getItem('uvus')?.toLocaleUpperCase() || '';
  }

  obtenerEstadisticas(id: number) {
    // 1. Recuperamos datos específicos del bloque (guardados en BloqueComponent)
    const total = Number(localStorage.getItem(`Total_B${id}`)) || 0;
    const aciertos = Number(localStorage.getItem(`Aciertos_B${id}`)) || 0;

    // 2. Cálculo de Éxito (% de aciertos sobre preguntas intentadas)
    const exito = total > 0 ? Math.round((aciertos / total) * 100) : 0;

    // 3. Cálculo de Progreso Visual (% sobre una meta de, por ejemplo, 20 aciertos)
    const metaAciertos = 20;
    const progreso = Math.min(Math.round((aciertos / metaAciertos) * 100), 100);

    // Devolvemos un objeto con todos los datos
    return {
      total: total,
      aciertos: aciertos,
      exito: exito,
      progreso: progreso,
    };
  }
}
