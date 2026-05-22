import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { AuthService } from '../../services/auth.service';


@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive],
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.css',
})
export class NavbarComponent {

  // LISTA DE BLOQUES DEL MENÚ
  // Cada bloque representa una parte del temario de la asignatura
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

  // DATOS DEL USUARIO LOGUEADO
  nombreUsuario: string | null = '';
  idUsuario: string | null = null;

  // ESTADO DE LOGIN (no se usa mucho porque lo controla el servicio)
  estaLogeado: boolean = false;

  // controla qué bloque está desplegado en el menú
  bloqueAbierto: number | null = null;

  // controla si el menú móvil está abierto o cerrado
  menuMovilAbierto: boolean = false;

  constructor(public authService: AuthService) {}

  ngOnInit() {

    // nos suscribimos al observable del usuario (login/logout en tiempo real)
    this.authService.uvus$.subscribe((uvus) => {

      // si hay usuario logueado
      if (uvus) {

        // formateamos el nombre (primera mayúscula)
        this.nombreUsuario =
          uvus.charAt(0).toUpperCase() + uvus.slice(1).toLowerCase();

        // obtenemos token guardado
        const token = this.authService.getToken();

        // si existe token, sacamos el id del usuario
        if (token) {
          try {
            this.idUsuario = this.authService.getUserId();
          } catch (e) {
            console.error('Error al decodificar el token', e);
          }
        }

      } else {
        // si no hay usuario logueado, limpiamos datos
        this.nombreUsuario = null;
        this.idUsuario = null;
      }
    });
  }

  // CERRAR SESIÓN
  logout() {
    this.authService.logout();
  }

  // ABRIR DESPLEGABLE DE UN BLOQUE (hover en desktop)
  abrirDesplegable(bloqueId: number) {
    this.bloqueAbierto = bloqueId;
  }

  // CERRAR DESPLEGABLE
  cerrarDesplegable() {
    this.bloqueAbierto = null;
  }

  // COMPROBAR SI UN BLOQUE ESTÁ ABIERTO
  estaAbierto(bloqueId: number): boolean {
    return this.bloqueAbierto === bloqueId;
  }

  // ABRIR / CERRAR MENÚ MÓVIL (hamburguesa)
  toggleMenuMovil() {
    this.menuMovilAbierto = !this.menuMovilAbierto;
  }

  // CERRAR MENÚ MÓVIL COMPLETAMENTE
  cerrarMenuMovil() {
    this.menuMovilAbierto = false;
    this.bloqueAbierto = null;
  }

  // TOGGLE DE DESPLEGABLE EN MÓVIL (click en bloque)
  toggleDesplegableMovil(bloqueId: number) {
    if (this.bloqueAbierto === bloqueId) {
      this.bloqueAbierto = null;
    } else {
      this.bloqueAbierto = bloqueId;
    }
  }
}