// importamos componentes básicos de angular
import { Component } from '@angular/core';

// módulo para usar directivas tipo ngIf, ngFor, etc
import { CommonModule } from '@angular/common';

// módulo para formularios con ngModel
import { FormsModule } from '@angular/forms';

// router para navegar entre páginas
import { Router, RouterLink } from '@angular/router';

// servicio de autenticación
import { AuthService } from '../../services/auth.service';

// interfaces para tipar los datos
import { Usuario } from '../../models/usuario';
import { Grado } from '../../models/grado';

@Component({
  selector: 'app-registro',
  standalone: true,

  // módulos que usa este componente
  imports: [CommonModule, FormsModule, RouterLink],

  // html y css del componente
  templateUrl: './registro.component.html',
  styleUrl: './registro.component.css',
})

export class RegistroComponent {

  // =========================
  // CAMPOS DEL FORMULARIO
  // =========================

  // nombre de usuario
  uvus: string = '';

  // correo
  email: string = '';

  // grado seleccionado
  grado: string = '';

  // contraseña
  password: string = '';



  // =========================
  // VARIABLES DE CONTROL
  // =========================

  // loading mientras se registra
  cargando: boolean = false;

  // error del uvus
  errorUvus: string = '';

  // error del email
  errorEmail: string = '';

  // error general
  error: string = '';



  // =========================
  // LISTA DE GRADOS
  // =========================

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



  // =========================
  // CONSTRUCTOR
  // =========================

  constructor(
    private authService: AuthService,
    private router: Router,
  ) {}



  // =========================
  // FUNCIÓN DE REGISTRO
  // =========================

  register() {

    // limpiamos errores antiguos
    this.errorUvus = '';
    this.errorEmail = '';
    this.error = '';



    // comprobamos que no haya campos vacíos
    if (!this.uvus || !this.email || !this.grado || !this.password) {

      this.error = 'Por favor, completa todos los campos';
      return;
    }



    // activamos loading
    this.cargando = true;



    // objeto que mandamos al backend
    const datos: Usuario = {

      uvus: this.uvus,
      email: this.email,
      grado: this.grado,
      password: this.password,
    };



    // guardamos el grado en localStorage
    localStorage.setItem('grado', this.grado);

    console.log('Datos enviados:', datos);



    // petición al backend
    this.authService.register(datos).subscribe({

      // registro correcto
      next: (response) => {

        console.log('Registro correcto', response);

        // redirigimos al inicio
        this.router.navigate(['/']);
      },



      // error en la petición
      error: (err) => {

        console.log(err);

        // quitamos loading
        this.cargando = false;



        // error 400 del backend
        if (err.status === 400) {

          const serverMessage = err.error;

          console.log('Error backend:', serverMessage);



          // uvus ya existe
          if (serverMessage.uvus) {

            this.errorUvus = 'El UVUS ya está registrado';
          }



          // error relacionado con email
          if (serverMessage.email) {

            const msgEmail =
              String(serverMessage.email).toLowerCase();



            // email inválido
            if (
              msgEmail.includes('valid') ||
              msgEmail.includes('formato')
            ) {

              this.errorEmail =
                'Introduce un correo válido';

            } else {

              // email repetido
              this.errorEmail =
                'Este email ya está en uso';
            }
          }

        } else {

          // error genérico
          this.error =
            'Error en el registro. Inténtalo otra vez.';
        }
      },
    });
  }
}