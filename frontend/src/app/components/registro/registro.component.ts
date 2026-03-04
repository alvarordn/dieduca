import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../services/auth.service';
// Importamos las interfaces
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
  // Inicializamos los campos siguiendo la interfaz Usuario
  uvus: string = '';
  email: string = '';
  grado: string = '';
  password: string = '';

  cargando: boolean = false;
  errorUvus: string = '';
  errorEmail: string = '';
  error: string = '';

  // Definimos la lista de grados usando la interfaz Grado
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
    { id: 3, nombre: 'Grado en Ingeniería Aeroespacial', valor: 'Grado en Ingeniería Aeroespacial' },
    { id: 4, nombre: 'Grado en Ingeniería Civil', valor: 'Grado en Ingeniería Civil' },
    { id: 5, nombre: 'Grado en Ingeniería Química', valor: 'Grado en Ingeniería Química' },
    {
      id: 6,
      nombre: 'Grado en Ingeniería de Organización Industrial',
      valor: 'Grado en Ingeniería de Organización Industrial',
    },
    { id: 7, nombre: 'Grado en Ingeniería de la Energía', valor: 'Grado en Ingeniería de la Energía' },
    {
      id: 8,
      nombre: 'Grado en Ingeniería Electrónica, Robótica y Mecatrónica',
      valor: 'Grado en Ingeniería Electrónica, Robótica y Mecatrónica',
    },
  ];

  constructor(
    private authService: AuthService,
    private router: Router,
  ) {}

  register() {
    this.errorUvus = '';
    this.errorEmail = '';

    if (!this.uvus || !this.email || !this.grado || !this.password) {
      this.error = 'Por favor, completa todos los campos';
      return;
    }

    this.cargando = true;

    // Creamos el objeto Usuario
    const datos: Usuario = {
      uvus: this.uvus,
      email: this.email,
      grado: this.grado,
      password: this.password,
    };
    localStorage.setItem('grado', this.grado);
    console.log('Datos a enviar:', datos);

    this.authService.register(datos).subscribe({
      next: (response) => {
        console.log('Registro exitoso', response);
        this.router.navigate(['/']);
      },
      error: (err) => {
        if (err.status === 400) {
          const serverMessage = err.error;
          console.log("Mensaje servidor: ",serverMessage);

          if (serverMessage.uvus) {
            this.errorUvus = 'El UVUS ya está registrado';
          }
          if(serverMessage.email){
            this.errorEmail = 'El email ya está registrado'
          }
        } else {
          this.error = 'Error en el registro. Por favor, inténtalo de nuevo.';
        }
        this.cargando = false;
      },
    });
  }
}
