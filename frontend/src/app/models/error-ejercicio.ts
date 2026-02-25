export interface ErrorEjercicio {
  id: number;
  bloqueId: number;
  fecha: Date;
  pregunta: string;
  respuestaDada: string;
  respuestaCorrecta: string;
  tipoCircuito: string; 
}