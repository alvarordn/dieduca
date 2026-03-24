import { Injectable } from '@angular/core';
import Swal from 'sweetalert2';
import { HttpClient } from '@angular/common/http';
// para hacer peticiones al backend
import { Observable, BehaviorSubject } from 'rxjs';
// Observable es la base de las peticiones asincronas, sin el no se podria hacer
// BehaviorSubject PERMITE EMITIR VALORES Y ALMACENAR EL ESTADO DE AUTENTICACION
import { tap } from 'rxjs/operators';
// aqui lo usamos para guardar el token, esta relacionado con el observable
import { Router } from '@angular/router';
import { Usuario } from '../models/usuario';
// Para contar los minutos, horas etc de cada usuario
import { interval, Subscription } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  private apiUrl = 'http://localhost:8000/api/auth';
  // url a la base de datos donde se le pasaran todas las peticiones de autenticacion

  private isAuthenticatedSubject = new BehaviorSubject<boolean>(
    this.hasToken(),
  );
  // como se dijo antes, behaviorSubject almacena el estado de autenticacion, mas concretamente en este caso un booleano, se inicializa con el resultado de hasToken()
  private timerSub?: Subscription;
  private isAuthenticated$ = this.isAuthenticatedSubject.asObservable();
  // Observable publico (de ahi el simbolo $), otros componentes pueden suscribirse para reaccionar a los cambios de autenticacion, como por ejemplo mostrar/ocultar botones de login o logout
  private uvusSubject = new BehaviorSubject<string | null>(
    sessionStorage.getItem('uvus'),
  );

  public uvus$ = this.uvusSubject.asObservable();
  constructor(
    private http: HttpClient,
    private router: Router,
  ) {}
  // inyecta las dependencias necesarias

  private hasToken(): boolean {
    return !!sessionStorage.getItem('token');
  }
  // funcion que le servira al behaviorSubject si el usuario esta autenticado o no, se usa el !! para convertir el resultado de (string o null) a (true o false)
getUserId(): string | null {
  const token: string | null = this.getToken();
  
  if (!token) {
    return null;
  }

  const partes: string[] = token.split('.');

  if (partes.length < 2) {
    return null;
  }

  try {
    const payloadBase64: string = partes[1];
    
    const payloadJson: string = atob(payloadBase64);
    const payload = JSON.parse(payloadJson);
    
    return payload.user_id ? String(payload.user_id) : null;
  } catch (e) {
    console.error("Error al decodificar el token:", e);
    return null;
  }
}

  login(uvus: string, password: string): Observable<any> {
    // se activa la secuencia de login
    return (
      this.http
        .post(`${this.apiUrl}/login/`, { uvus, password })
        // realiza una peticion post a /api/auth/login, con las credenciales enviadas, devuelve un observable
        .pipe(
          tap((response: any) => {
            // Utiliza el operador tap para ejecutar código inmediatamente después de que la petición sea exitosa, pero antes de que el componente que llama reciba la respuesta
            if (response.token) {
              sessionStorage.setItem('token', response.token);
              sessionStorage.setItem('uvus', response.uvus);
              // si el login es exitoso, se guarda en local para que persista la sesion
              this.uvusSubject.next(response.uvus);
              this.isAuthenticatedSubject.next(true);
              // Notifica a todos los suscriptores (como el AuthGuard o la barra de navegación) que el usuario ahora está autenticado, cambiando el estado global
            }
          }),
        )
    );
  }

  register(datos: Usuario): Observable<Usuario> {
    return this.http.post(`${this.apiUrl}/register/`, datos).pipe(
      tap((response: any) => {
        if (response.token) {
          sessionStorage.setItem('token', response.token);
          sessionStorage.setItem('uvus', response.uvus);
          // Actualiza el valor del UVUS en el BehaviorSubject
          this.uvusSubject.next(response.uvus);
          this.isAuthenticatedSubject.next(true);
        }
      }),
    );
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
      if (result.isConfirmed) {
        sessionStorage.removeItem('token');
        sessionStorage.removeItem('uvus');
        this.uvusSubject.next(null);
        this.isAuthenticatedSubject.next(false);
        this.router.navigate(['/']);

        Swal.fire({
          title: 'Sesión cerrada',
          icon: 'success',
          timer: 1500,
          showConfirmButton: false,
          padding: '1.5rem',
          didOpen: (popup) => {
            popup.style.borderRadius = '24px';
          },
        });
      }
    });
  }
  // cierra sesion, se eliminan ambos campos del local
  // redirige al login

  startTracking(username: String) {
    this.timerSub = interval(60000).subscribe(() => {
      this.http
        .post('http://localhost:8000/api/auth/track-time/', { uvus: username })
        .subscribe({
          error: (err) => console.error('error al registrar'),
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
  // Recupera el valor del token almacenado en local storage, en este caso si es string o nulo

  getUvus(): string | null {
    return sessionStorage.getItem('uvus');
  }
  // Recupera el campo uvus almacenado de igual forma

  isLoggedIn(): boolean {
    return this.hasToken();
  }
  // Metodo que expone el estado de autenticacion basado en la existencia del token (hasToken deuvelve true o false)
}

// Observable

// Definicion:
// Una fuente de datos que puede emitir cero o más valores.
// Es un patrón de diseño que define un productor de datos
// (Observable) y un consumidor de datos (Observer/Subscriber).

// Las llamadas a this.http.post(...) devuelven Observables,
// ya que la respuesta del servidor (éxito o error) llegará en
// el futuro.

// Son perezosos (lazy), lo que significa que el código dentro
// del Observable (la petición HTTP) no se ejecuta hasta que
// alguien se suscribe a él.

// En login.component.ts, la petición
// no se envía a Django hasta que llamas a .subscribe() en el
// Observable devuelto por authService.login().
