import { HttpInterceptorFn, HttpErrorResponse } from '@angular/common/http';
import { inject } from '@angular/core';
import { AuthService } from '../services/auth.service';
import { catchError, throwError } from 'rxjs';
import Swal from 'sweetalert2';

export const authTokenInterceptor: HttpInterceptorFn = (req, next) => {
  const authService = inject(AuthService);
  const token = authService.getToken();

  let authReq = req;

  // 1. SI HAY TOKEN, LO PONEMOS EN LA CABECERA
  if (token) {
    authReq = req.clone({
      headers: req.headers.set('Authorization', `Bearer ${token}`)
    });
  }

  // 2. ESCUCHAMOS LA RESPUESTA PARA DETECTAR SI EL TOKEN CADUCA
  return next(authReq).pipe(
    catchError((error: HttpErrorResponse) => {
      // 401 = No autorizado (Token caducado)
      // 403 = Prohibido (Token inválido)
      if (error.status === 401 || error.status === 403) {
        console.warn("Sesión caducada. Expulsando usuario...");

        // Llamamos a la limpieza inmediata sin preguntas
        authService.forceLogout();

        // Avisamos al usuario de forma elegante
        Swal.fire({
          title: 'Sesión caducada',
          text: 'Por seguridad, tu sesión ha finalizado. Por favor, entra de nuevo.',
          icon: 'info',
          confirmButtonColor: '#3b82f6',
          timer: 4000,
          didOpen: (popup) => { popup.style.borderRadius = '24px'; }
        });
      }
      return throwError(() => error);
    })
  );
};