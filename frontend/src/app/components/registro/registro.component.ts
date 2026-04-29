import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../services/auth.service';
// Importamos las interfaces que vamos a usar para tipar datos
import { Usuario } from '../../models/usuario';
import { Grado } from '../../models/grado';

@Component({
  selector: 'app-registro',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './registro.component.html',
  styleUrl: './registro.component.css',
})
export class RegistroComponent {

  // Campos del formulario de registro
  // Se inicializan vacíos porque el usuario los va a rellenar
  uvus: string = '';
  email: string = '';
  grado: string = '';
  password: string = '';

  // Variables de control del estado de la UI
  cargando: boolean = false; // para mostrar loading mientras se registra
  errorUvus: string = ''; // error específico del UVUS
  errorEmail: string = ''; // error específico del email
  error: string = ''; // error general del formulario

  // Lista de grados disponibles para el select
  // Usamos la interfaz Grado para mantener tipado correcto
  grados: Grado[] = [
    {
      id: 1,
      nombre: 'Grado de Ingeniería de Tecnologías Industriales',
      valor: 'Grado de Ingeniería de Tecnologías Industriales',
    },
    {
      id: 2,
      nombre: 'Grado en Ingeniería de las Tecnologías de Telecomunicación',
      valor: 'Grado en Ingeniería de las Tecnologías de Telecomunicación',
    },
    {
      id: 3,
      nombre: 'Grado en Ingeniería Aeroespacial',
      valor: 'Grado en Ingeniería Aeroespacial',
    },
    {
      id: 4,
      nombre: 'Grado en Ingeniería Civil',
      valor: 'Grado en Ingeniería Civil',
    },
    {
      id: 5,
      nombre: 'Grado en Ingeniería Química',
      valor: 'Grado en Ingeniería Química',
    },
    {
      id: 6,
      nombre: 'Grado en Ingeniería de Organización Industrial',
      valor: 'Grado en Ingeniería de Organización Industrial',
    },
    {
      id: 7,
      nombre: 'Grado en Ingeniería de la Energía',
      valor: 'Grado en Ingeniería de la Energía',
    },
    {
      id: 8,
      nombre: 'Grado en Ingeniería Electrónica, Robótica y Mecatrónica',
      valor: 'Grado en Ingeniería Electrónica, Robótica y Mecatrónica',
    },
  ];

  // Inyectamos servicios:
  // authService -> para llamar al backend
  // router -> para navegar entre páginas
  constructor(
    private authService: AuthService,
    private router: Router,
  ) {}

  // Función principal de registro
  register() {

    // Reseteamos errores cada vez que se intenta registrar
    // así evitamos que se queden mensajes antiguos
    this.errorUvus = '';
    this.errorEmail = '';

    // Validación básica: comprobar que no hay campos vacíos
    if (!this.uvus || !this.email || !this.grado || !this.password) {
      this.error = 'Por favor, completa todos los campos';
      return;
    }

    // Activamos loading mientras se hace la petición
    this.cargando = true;

    // Creamos el objeto que se enviará al backend
    const datos: Usuario = {
      uvus: this.uvus,
      email: this.email,
      grado: this.grado,
      password: this.password,
    };

    // Guardamos el grado en localStorage (por si lo usamos después)
    localStorage.setItem('grado', this.grado);

    console.log('Datos a enviar:', datos);

    // Llamada al servicio de registro (API)
    this.authService.register(datos).subscribe({

      // Si todo va bien
      next: (response) => {
        console.log('Registro exitoso', response);

        // Redirigimos al login o home
        this.router.navigate(['/']);
      },

      // Si hay error en la petición
      error: (err) => {

        // Reiniciamos loading
        this.cargando = false;

        // Error de validación del backend
        if (err.status === 400) {
          const serverMessage = err.error;
          console.log('Mensaje servidor: ', serverMessage);

          // Error UVUS duplicado
          if (serverMessage.uvus) {
            this.errorUvus = 'El UVUS ya está registrado';
          }

          // Error de email
          if (serverMessage.email) {
            const msgEmail = String(serverMessage.email).toLowerCase();

            // Si el backend dice que el formato es incorrecto
            if (msgEmail.includes('valid') || msgEmail.includes('formato')) {
              this.errorEmail = 'Introduce un correo electrónico válido';
            } else {
              // Si ya existe en la base de datos
              this.errorEmail = 'Este email ya está en uso';
            }
          }
        } else {
          // Error genérico si falla algo raro
          this.error = 'Error en el registro. Por favor, inténtalo de nuevo.';
        }
      },
    });
  }
}