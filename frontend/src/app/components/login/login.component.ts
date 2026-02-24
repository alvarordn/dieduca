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
  uvus: string = '';
  password: string = '';
  cargando: boolean = false;
  error: string = '';

  constructor(
    private authService: AuthService,
    private router: Router,
  ) {}

  login() {
    // metodo que se ejecuta cuando se envia el login
    this.error = '';

    if (!this.uvus || !this.password) {
      this.error = 'Por favor, completa todos los campos';
      return;
    }

    Swal.fire({
      title: 'Autenticando...',
      text: 'Conectando con el servidor de Ingeniería',
      allowOutsideClick: false,
      showConfirmButton: false,
      didOpen: () => {
        Swal.showLoading();
        const popup = Swal.getPopup();
        if (popup) popup.style.borderRadius = '24px';
      },
    });

    this.authService.login(this.uvus, this.password).subscribe({
      // Llama al método login del AuthService y se suscribe al
      // Observable devuelto, iniciando la petición HTTP a Django
      next: (response) => {
        Swal.close();
        this.router.navigate(['/']);
      },
      error: (err) => {
        Swal.fire({
          title: 'Error de acceso',
          text: 'UVUS o contraseña incorrectos.',
          icon: 'error',
          confirmButtonColor: '#3b82f6',
          confirmButtonText: 'Reintentar',
          didOpen: (popup) => {
            popup.style.borderRadius = '24px';
          },
        });
        this.error = 'UVUS o contraseña incorrectos.';

        this.cargando = false;
      },
    });
  }
}
