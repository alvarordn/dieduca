import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class CircuitosService {
  private apiUrl = 'http://localhost:8000/api/circuitos';

  constructor(private http: HttpClient) { }

  generarCircuito(datos: any): Observable<any> {
    const token = localStorage.getItem('access_token'); 

    const headers = new HttpHeaders({
      'Authorization': `Bearer ${token}`
    });

    return this.http.post(`${this.apiUrl}/generar-circuito/`, datos, { headers });
  }

  testConnection(): Observable<any> {
    return this.http.get(`${this.apiUrl}/test/`);
  }
}