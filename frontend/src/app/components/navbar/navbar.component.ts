import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule, RouterLink],
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
    { id: 6, nombre: 'Bloque 6' },
    { id: 7, nombre: 'Bloque 7' },
    { id: 8, nombre: 'Bloque 8' },
    { id: 9, nombre: 'Bloque 9' },
    { id: 10, nombre: 'Bloque 10' },
    { id: 11, nombre: 'Bloque 11' },

   
  ];
  nombreUsuario: string | null = '';
  idUsuario: string | null = null;
  estaLogeado: boolean = false;
  bloqueAbierto: number | null = null;
 
  
  menuMovilAbierto: boolean = false;

  constructor(public authService: AuthService) {}

  ngOnInit() {
    this.authService.uvus$.subscribe((uvus) => {
      if (uvus) {
        this.nombreUsuario =
          uvus.charAt(0).toUpperCase() + uvus.slice(1).toLowerCase();

 
        const token = localStorage.getItem('token');
        console.log('Token almacenado:', token); 
        if (token) {
          try {
          
            const payload = JSON.parse(atob(token.split('.')[1]));
            console.log('Payload decodificado:', payload);
            this.idUsuario = payload.user_id; 
            console.log("ID del Usuario", this.idUsuario)
          } catch (e) {
            console.error('Error al decodificar el token', e);
          }
        }
      } else {
        this.nombreUsuario = null;
        this.idUsuario = null;
      }
    });
  }

  logout() {
    this.authService.logout();
  }

  abrirDesplegable(bloqueId: number) {
    this.bloqueAbierto = bloqueId;
  }



  cerrarDesplegable() {
    this.bloqueAbierto = null;
  }



  estaAbierto(bloqueId: number): boolean {
    return this.bloqueAbierto === bloqueId;
  }


  toggleMenuMovil() {
    this.menuMovilAbierto = !this.menuMovilAbierto;
  }



  cerrarMenuMovil() {
    this.menuMovilAbierto = false;
    this.bloqueAbierto = null;
  }


  toggleDesplegableMovil(bloqueId: number) {
    if (this.bloqueAbierto === bloqueId) {
      this.bloqueAbierto = null;
    } else {
      this.bloqueAbierto = bloqueId;
    }
  }


}
