import { Injectable } from '@angular/core';
import Swal from 'sweetalert2';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject, Subscription, interval } from 'rxjs';
import { tap } from 'rxjs/operators';
import { Router } from '@angular/router';
import { Usuario } from '../models/usuario';

@Injectable({
  providedIn: 'root',
})
export class AuthService {

  private apiUrl = 'http://localhost:8000/api/auth';

  private isAuthenticatedSubject = new BehaviorSubject<boolean>(this.hasToken());
  public isAuthenticated$ = this.isAuthenticatedSubject.asObservable();

  private uvusSubject = new BehaviorSubject<string | null>(sessionStorage.getItem('uvus'));
  public uvus$ = this.uvusSubject.asObservable();

  private timerSub?: Subscription;

  constructor(private http: HttpClient, private router: Router) {}

  private hasToken(): boolean {
    return !!sessionStorage.getItem('token');
  }

  getUserId(): string | null {

    const token = this.getToken();
    if (!token) return null;

    try {

      const partes = token.split('.');
      if (partes.length < 2) return null;

      const payloadJson = atob(partes[1]);
      const payload = JSON.parse(payloadJson);

      // comprobación estricta (sin ==)
      return payload.user_id ? String(payload.user_id) : null;

    } catch (e) {

      console.error("Error al decodificar el token:", e);
      return null;
    }
  }

  forceLogout() {

    this.stopTracking();

    sessionStorage.removeItem('token');
    sessionStorage.removeItem('uvus');

    this.uvusSubject.next(null);
    this.isAuthenticatedSubject.next(false);

    this.router.navigate(['/login']);
  }

  logout() {

    Swal.fire({
      title: '¿Cerrar sesión?',
      text: '¿Estás seguro de que quieres salir?',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonText: 'Sí, cerrar sesión',
      cancelButtonText: 'Cancelar',
      confirmButtonColor: '#3b82f6',
      cancelButtonColor: '#64748b',
      didOpen: (popup) => {
        popup.style.borderRadius = '24px';
      },
    }).then((result) => {

      // aquí antes NO hay ==, usamos propiedad booleana directa
      if (result.isConfirmed) {

        this.forceLogout();

        Swal.fire({
          title: 'Sesión cerrada',
          icon: 'success',
          timer: 1500,
          showConfirmButton: false,
          didOpen: (popup) => {
            popup.style.borderRadius = '24px';
          },
        });
      }
    });
  }

  login(uvus: string, password: string): Observable<any> {

    return this.http.post(`${this.apiUrl}/login/`, { uvus, password }).pipe(

      tap((response: any) => {

        if (response.token) {

          sessionStorage.setItem('token', response.token);
          sessionStorage.setItem('uvus', response.uvus);

          this.uvusSubject.next(response.uvus);
          this.isAuthenticatedSubject.next(true);
        }
      })
    );
  }

  register(datos: Usuario): Observable<Usuario> {

    return this.http.post(`${this.apiUrl}/register/`, datos).pipe(

      tap((response: any) => {

        if (response.token) {

          sessionStorage.setItem('token', response.token);
          sessionStorage.setItem('uvus', response.uvus);

          this.uvusSubject.next(response.uvus);
          this.isAuthenticatedSubject.next(true);
        }
      })
    );
  }

  startTracking(username: string) {

    this.timerSub = interval(60000).subscribe(() => {

      this.http.post(`${this.apiUrl}/track-time/`, { uvus: username })
        .subscribe({
          error: () => console.error('Error al registrar tiempo')
        });
    });
  }

  stopTracking() {
    if (this.timerSub) {
      this.timerSub.unsubscribe();
    }
  }

  getToken(): string | null {
    return sessionStorage.getItem('token');
  }

  getUvus(): string | null {
    return sessionStorage.getItem('uvus');
  }

  isLoggedIn(): boolean {
    return this.hasToken();
  }
}