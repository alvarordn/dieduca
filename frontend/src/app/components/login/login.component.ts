/* componente de login de la aplicación */
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import Swal from 'sweetalert2';

@Component({
  selector: 'app-login',
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './login.component.html',
  styleUrl: './login.component.css',
})
export class LoginComponent {

  // usuario (UVUS) que introduce el alumno
  uvus: string = '';

  // contraseña del usuario
  password: string = '';

  // controla el estado de carga del login
  cargando: boolean = false;

  // mensaje de error que se muestra en pantalla
  error: string = '';

  constructor(
    private authService: AuthService,
    private router: Router,
  ) {}

  // método que se ejecuta al enviar el formulario de login
  login() {

    // resetea errores anteriores
    this.error = '';

    // validación básica de campos vacíos
    if (!this.uvus || !this.password) {
      this.error = 'Por favor, completa todos los campos';
      return;
    }

    // popup de SweetAlert mientras se autentica
    Swal.fire({
      title: 'Autenticando...',
      text: 'Conectando con el servidor de Ingeniería',
      allowOutsideClick: false,
      showConfirmButton: false,

      // se ejecuta cuando el popup se abre
      didOpen: () => {
        Swal.showLoading();

        // estilo visual del popup
        const popup = Swal.getPopup();
        if (popup) popup.style.borderRadius = '24px';
      },
    });

    // llamada al backend mediante el servicio de auth
    this.authService.login(this.uvus, this.password).subscribe({

      // si el login es correcto
      next: (response) => {

        // cierra el loading
        Swal.close();

        // redirige a la página principal
        this.router.navigate(['/']);
      },

      // si hay error en el login
      error: (err) => {

        // muestra alerta de error
        Swal.fire({
          title: 'Error de acceso',
          text: 'UVUS o contraseña incorrectos.',
          icon: 'error',
          confirmButtonColor: '#3b82f6',
          confirmButtonText: 'Reintentar',

          // estilo del popup
          didOpen: (popup) => {
            popup.style.borderRadius = '24px';
          },
        });

        // guarda el mensaje de error para mostrarlo en el HTML
        this.error = 'UVUS o contraseña incorrectos.';

        // desactiva estado de carga
        this.cargando = false;
      },
    });
  }
}