import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { CircuitosService } from '../../services/circuitos.service';
import { CircuitViewerComponent } from '../circuit-viewer/circuit-viewer.component';


@Component({
  selector: 'app-bloque',
  standalone: true,
  imports: [CommonModule, FormsModule, CircuitViewerComponent],
  templateUrl: './bloque.component.html',
  styleUrl: './bloque.component.css'
})
export class BloqueComponent implements OnInit {
  bloqueId: string = '';
  circuitoGenerado: any = null;
  cargando: boolean = false;
  errorCircuito: string = '';
  
  // parametros predeterminados del circuito
  rows: number = 2;
  cols: number = 3;

    respuestas = {
    pregunta1: null as number | null,
    pregunta2: null as number | null,
    pregunta3: null as number | null
  };
  
  // Control del resultado
  resultadoVisible: boolean = false;
  mensajeResultado: string = '';

  constructor(
    private route: ActivatedRoute, // para leer la URL
    private circuitosService: CircuitosService // para hacer peticiones a Django.
  ) {}

  ngOnInit() { // se ejecuta cada vez que se crea el componente
    this.route.params.subscribe(params => { // Subscribe seria como para que avisase de que se actualiza un valor
      this.bloqueId = params['id']; // si se va a /bloque/3, params['id']
    });
  }



  generarEjercicio() {
    this.cargando = true; // activa el cargando
    this.circuitoGenerado = null; // limpia el circuito anterior

    this.limpiarRespuestas();
    
    const datos = { // crea el objeto
      bloque: this.bloqueId,
      rows: this.rows,
      cols: this.cols
    };
    
    this.circuitosService.generarCircuito(datos) // llama al servicio (que devuelve un observable) 
      .subscribe({
        next: (respuesta) => { // se ejecuta cuando django responde bien
          console.log('Respuesta del servidor:', respuesta);
          this.circuitoGenerado = respuesta;
          this.cargando = false;
        },
        error: (error) => { // se ejecuta si algo falla
          console.error('Error al generar circuito:', error);
          this.errorCircuito = 'Error al generar el circuito. Debe Iniciar Sesion primero.';
          this.cargando = false;
        }
      });
  }


comprobarRespuestas() {
  if (this.respuestas.pregunta1 === null || 
      this.respuestas.pregunta2 === null || 
      this.respuestas.pregunta3 === null) {
    this.mensajeResultado = 'Por favor, rellena todas las preguntas';
    this.resultadoVisible = true;
    return;
  }

  this.mensajeResultado = 'Comprobando respuesta';
  this.resultadoVisible = true;
}

limpiarRespuestas() {
  this.respuestas = {
    pregunta1: null,
    pregunta2: null,
    pregunta3: null
  };
  this.resultadoVisible = false;
  this.mensajeResultado = '';
}
}