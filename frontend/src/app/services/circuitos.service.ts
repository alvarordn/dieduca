import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class CircuitosService {
  // Base de la API para circuitos
  private baseUrl = 'http://localhost:8000/api/circuitos';

  constructor(private http: HttpClient) { }

  generarCircuito(datos: any): Observable<any> {
    // Nota: Hemos quitado los headers de Authorization porque usamos AllowAny en el backend
    // Esto evita errores si el token de la sesión ha expirado.
    return this.http.post(`${this.baseUrl}/generar-circuito/`, datos).pipe(
      map((res: any) => {
        // Aprovechamos para limpiar los potenciales complejos (46.15+0j -> 46.15)
        if (res.success && res.circuito) {
          res.circuito.nodos.forEach((n: any) => {
            n.potential = n.potential.replace(/\+0j|\(|\)/g, '');
          });
        }
        return res;
      })
    );
  }

  testConnection(): Observable<any> {
    // La ruta correcta según tu urls.py
    return this.http.get(`${this.baseUrl}/test/`);
  }
}