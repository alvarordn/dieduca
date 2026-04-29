import { Component } from '@angular/core';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-home',
  imports: [],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css',
})
export class HomeComponent {

  // nombre del usuario (se muestra en la home si está logueado)
  nombreUsuario: string | null = null;

  // grado del usuario (guardado en localStorage)
  gradoUsuario: string | null = null;

  constructor(private service: AuthService) {}

  ngOnInit() {

    // saco el grado del usuario desde localStorage
    this.gradoUsuario = localStorage.getItem('grado');

    // me suscribo al observable para saber si hay usuario logueado
    this.service.uvus$.subscribe((uvus) => {

      // si hay usuario logueado
      if (uvus) {

        // formateo el nombre (primera letra mayúscula, resto minúscula)
        this.nombreUsuario =
          uvus.charAt(0).toUpperCase() + uvus?.slice(1).toLocaleLowerCase();

      } else {
        // si no hay usuario, limpio el nombre
        this.nombreUsuario = null;
      }

    });

    // debug: muestra el usuario en consola
    console.log('UVUS del usuario:', this.nombreUsuario);
  }
}