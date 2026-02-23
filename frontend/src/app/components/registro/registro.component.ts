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
  styleUrl: './registro.component.css'
})
export class RegistroComponent {
  // Inicializamos los campos siguiendo la interfaz Usuario
  uvus: string = '';
  email: string = '';
  grado: string = ''; 
  password: string = '';
  
  cargando: boolean = false;
  error: string = '';

  // Definimos la lista de grados usando la interfaz Grado
  grados: Grado[] = [
    { id: 1, nombre: 'Grado de Ingeniería de Tecnologías Industriales', valor: 'GITI' },
    { id: 2, nombre: 'Grado en Ingeniería de las Tecnologías de Telecomunicación', valor: 'GITT' },
    { id: 3, nombre: 'Grado en Ingeniería Aeroespacial', valor: 'GIA' },
    { id: 4, nombre: 'Grado en Ingeniería Civil', valor: 'GIC' },
    { id: 5, nombre: 'Grado en Ingeniería Química', valor: 'GIQ' },
    { id: 6, nombre: 'Grado en Ingeniería de Organización Industrial', valor: 'GIOI' },
    { id: 7, nombre: 'Grado en Ingeniería de la Energía', valor: 'GIE' },
    { id: 8, nombre: 'Grado en Ingeniería Electrónica, Robótica y Mecatrónica', valor: 'GIERM' }
  ];

  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  register() {
    this.error = '';

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
      password: this.password
    };

    this.authService.register(datos).subscribe({
      next: (response) => {
        console.log('Registro exitoso', response);
        this.router.navigate(['/']);
      },
      error: (err) => {
        if (err.status === 400) {
          this.error = 'El UVUS o el correo ya están registrados';
        } else {
          this.error = 'Error en el registro. Por favor, inténtalo de nuevo.';
        }
        console.error('Error en el registro:', err);
        this.cargando = false;
      }
    });
  }
}