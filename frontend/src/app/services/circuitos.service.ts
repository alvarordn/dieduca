import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class CircuitosService {

  // URL base apuntando al prefijo del servidor Django local
  private baseUrl = 'http://127.0.0.1:8000/api/circuitos';

  constructor(private http: HttpClient) {}

  /**
   * Envía los parámetros al backend por POST para generar el circuito correspondiente.
   * Filtra e inicializa inconsistencias en las fases complejas si es necesario.
   */
  generarCircuito(datos: any): Observable<any> {
    return this.http.post(`${this.baseUrl}/generar_circuito/`, datos).pipe(
      map((res: any) => {
        // Validación estricta del árbol de propiedades del JSON de respuesta
        const tieneCircuito =
          res &&
          res.success &&
          res.circuito &&
          res.circuito.sections;

        if (tieneCircuito) {
          res.circuito.sections.forEach((s: any) => {
            // Inicialización preventiva de valores imaginarios indefinidos
            if (s?.Z_phase?.im === undefined) {
              s.Z_phase = s.Z_phase || {};
              s.Z_phase.im = 0;
            }
          });
        }
        return res;
      }),
    );
  }
}