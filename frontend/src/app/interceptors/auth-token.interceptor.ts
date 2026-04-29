import { HttpInterceptorFn, HttpErrorResponse } from '@angular/common/http';
import { inject } from '@angular/core';
import { AuthService } from '../services/auth.service';
import { catchError, throwError } from 'rxjs';
import Swal from 'sweetalert2';

export const authTokenInterceptor: HttpInterceptorFn = (req, next) => {

  // Inyectamos el servicio de autenticación dentro del interceptor
  const authService = inject(AuthService);

  // Obtenemos el token guardado (si existe)
  const token = authService.getToken();

  // Variable que usaremos para modificar la request
  let authReq = req;

  // 1. SI EXISTE TOKEN LO AÑADIMOS A LA CABECERA
  if (token) {

    // Clonamos la request original para no modificarla directamente
    authReq = req.clone({
      headers: req.headers.set('Authorization', `Bearer ${token}`)
    });
  }

  // 2. INTERCEPTAMOS RESPUESTAS PARA CONTROLAR ERRORES DE AUTH
  return next(authReq).pipe(

    catchError((error: HttpErrorResponse) => {

      // 401 o 403 significa que la sesión no es válida
      if (error.status == 401 || error.status == 403) {

        console.warn("Sesión caducada. Expulsando usuario...");

        // Cerramos sesión y limpiamos datos locales
        authService.forceLogout();

        // Mostramos alerta al usuario
        Swal.fire({

          title: 'Sesión caducada',
          text: 'Por seguridad, tu sesión ha finalizado. Por favor, entra de nuevo.',
          icon: 'info',
          confirmButtonColor: '#3b82f6',
          timer: 4000,

          // Ajuste visual del modal
          didOpen: (popup) => {
            popup.style.borderRadius = '24px';
          }
        });
      }

      // Reenviamos el error para que siga su flujo normal
      return throwError(() => error);
    })
  );
};