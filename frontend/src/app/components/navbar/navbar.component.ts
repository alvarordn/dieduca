// importamos lo necesario de angular
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink, RouterLinkActive } from '@angular/router';

// servicio que controla login, logout y datos del usuario
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-navbar',
  standalone: true,

  // módulos que usa este componente
  imports: [CommonModule, RouterLink, RouterLinkActive],

  // html y css del componente
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.css',
})

export class NavbarComponent {

  // lista de bloques que salen en el menú
  // cada uno tiene un id y un nombre
  bloques = [
    { id: 1, nombre: 'Bloque 1' },
    { id: 2, nombre: 'Bloque 2' },
    { id: 3, nombre: 'Bloque 3' },
    { id: 4, nombre: 'Bloque 4' },
    { id: 5, nombre: 'Bloque 5' },
    { id: 6, nombre: 'Bloque 6' },
    { id: 7, nombre: 'Bloque 7' },
    { id: 8, nombre: 'Bloque 8' },
    { id: 9, nombre: 'Bloque 9' },
    { id: 10, nombre: 'Bloque 10' },
  ];

  // nombre del usuario que ha iniciado sesión
  nombreUsuario: string | null = null;

  // id del usuario sacado del token
  idUsuario: string | null = null;

  // bloque desplegable abierto
  bloqueAbierto: number | null = null;

  // controla si el menú móvil está abierto
  menuMovilAbierto = false;

  // inyectamos el servicio de autenticación
  constructor(public authService: AuthService) {}

  // se ejecuta cuando carga el componente
  ngOnInit() {

    // escuchamos cambios del usuario
    this.authService.uvus$.subscribe((uvus) => {

      // si existe usuario
      if (uvus) {

        // ponemos la primera letra en mayúscula
        this.nombreUsuario =
          uvus.charAt(0).toUpperCase() +
          uvus.slice(1).toLowerCase();

        // obtenemos el token guardado
        const token = this.authService.getToken();

        // si hay token intentamos sacar el id
        if (token) {
          try {
            this.idUsuario = this.authService.getUserId();

          } catch (error) {

            // error por si el token falla
            console.error('Error leyendo el token', error);
          }
        }

      } else {

        // si no hay usuario limpiamos datos
        this.nombreUsuario = null;
        this.idUsuario = null;
      }
    });
  }

  // cerrar sesión
  logout() {
    this.authService.logout();
  }

  // abrir desplegable de un bloque
  abrirDesplegable(id: number) {
    this.bloqueAbierto = id;
  }

  // cerrar desplegable
  cerrarDesplegable() {
    this.bloqueAbierto = null;
  }

  // comprobar si un bloque está abierto
  estaAbierto(id: number): boolean {
    return this.bloqueAbierto === id;
  }

  // abrir o cerrar menú móvil
  toggleMenuMovil() {
    this.menuMovilAbierto = !this.menuMovilAbierto;
  }

  // cerrar menú móvil entero
  cerrarMenuMovil() {
    this.menuMovilAbierto = false;
    this.bloqueAbierto = null;
  }

  // abrir/cerrar desplegable en móvil
  toggleDesplegableMovil(id: number) {

    // si ya está abierto se cierra
    if (this.bloqueAbierto === id) {
      this.bloqueAbierto = null;

    } else {

      // si no, se abre
      this.bloqueAbierto = id;
    }
  }
}