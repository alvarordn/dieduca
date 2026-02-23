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
  bloques = [
    { id: 1, nombre: 'Bloque 1' },
    { id: 2, nombre: 'Bloque 2' },
    { id: 3, nombre: 'Bloque 3' },
    { id: 4, nombre: 'Bloque 4' },
    { id: 5, nombre: 'Bloque 5' },
  ];
  nombreUsuario: string | null = '';  

  constructor(public authService: AuthService) {}

  ngOnInit() {
    this.nombreUsuario = this.authService.getUvus();
    console.log('UVUS del usuario:', this.nombreUsuario);
  }

  logout() {
      this.authService.logout();
  }

  bloqueAbierto: number | null = null;
  // bloqueAbierto se le asigna el valor nulo predeterminadamente y cuando no se pasa el raton por encima de ningun bloque

  menuMovilAbierto: boolean = false;

  abrirDesplegable(bloqueId: number) {
    this.bloqueAbierto = bloqueId;
  }

  // al hacer hover sobre algun elemento del navbar se ejecuta esta funcion la cual se le pasa el id del elemento del navbar y por lo tanto bloqueAbierto deja de ser null

  cerrarDesplegable() {
    this.bloqueAbierto = null;
  }

  // cuando se quita el hover inmediatamente se cambia null el valor

  estaAbierto(bloqueId: number): boolean {
    return this.bloqueAbierto === bloqueId;
  }

  // si el id del bloque coincide con su id se abre el menu desplegable

  toggleMenuMovil() {
    this.menuMovilAbierto = !this.menuMovilAbierto;
  }

  // cambia el valor booleano de menuMovilAbierto, abre o cierra el menu hamburguesa

  cerrarMenuMovil() {
    this.menuMovilAbierto = false;
    this.bloqueAbierto = null;
  }

  // comprueba que se ha cerrado el menu hamburguesa y tambien que no haya ningun desplegable del bloque se quede abierto

  toggleDesplegableMovil(bloqueId: number) {
    if (this.bloqueAbierto === bloqueId) {
      this.bloqueAbierto = null;
    } else {
      this.bloqueAbierto = bloqueId;
    }
  }

  // toggleDesplegableMovil: gestiona la apertura y cierre del desplegable de bloques en la vista movil
  // si el desplegable ya estaba abierto (coincide el ID), lo cierra (lo pone a null)
  // si estaba cerrado, lo abre (asigna el nuevo ID). Esto permite abrir y cerrar con un solo clic
}
