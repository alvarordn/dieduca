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

  private isAuthenticated$ = this.isAuthenticatedSubject.asObservable();
  // Observable publico (de ahi el simbolo $), otros componentes pueden suscribirse para reaccionar a los cambios de autenticacion, como por ejemplo mostrar/ocultar botones de login o logout
  private uvusSubject = new BehaviorSubject<string | null>(
    localStorage.getItem('uvus'),
  );
  public uvus$ = this.uvusSubject.asObservable();
  constructor(
    private http: HttpClient,
    private router: Router,
  ) {}
  // inyecta las dependencias necesarias

  private hasToken(): boolean {
    return !!localStorage.getItem('token');
  }
  // funcion que le servira al behaviorSubject si el usuario esta autenticado o no, se usa el !! para convertir el resultado de (string o null) a (true o false)

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
              localStorage.setItem('token', response.token);
              localStorage.setItem('uvus', response.uvus);
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
          localStorage.setItem('token', response.token);
          localStorage.setItem('uvus', response.uvus);
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
      // Inyectamos el estilo directamente al abrirse
      didOpen: (popup) => {
        popup.style.borderRadius = '24px';
      },
    }).then((result) => {
      if (result.isConfirmed) {
        localStorage.removeItem('token');
        localStorage.removeItem('uvus');
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

  getToken(): string | null {
    return localStorage.getItem('token');
  }
  // Recupera el valor del token almacenado en local storage, en este caso si es string o nulo

  getUvus(): string | null {
    return localStorage.getItem('uvus');
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
