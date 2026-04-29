import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class CircuitosService {

  // URL base del backend de circuitos
  private baseUrl = 'http://127.0.0.1:8000/api/circuitos';

  constructor(private http: HttpClient) {}

  // Genera un circuito y limpia datos inconsistentes del backend
  generarCircuito(datos: any): Observable<any> {

    return this.http.post(`${this.baseUrl}/generar_circuito/`, datos).pipe(

      map((res: any) => {

        console.log('backend', res);

        // Comprobamos que la respuesta es válida sin comparaciones débiles
        const tieneCircuito =
          res &&
          res.success &&
          res.circuito &&
          res.circuito.sections;

        if (tieneCircuito) {

          res.circuito.sections.forEach((s: any) => {

            // Si no existe la parte imaginaria, la inicializamos a 0
            if (s?.Z_phase?.im === undefined) {
              s.Z_phase.im = 0;
            }
          });
        }

        return res;
      }),
    );
  }
}