import { Component } from '@angular/core';
import { AuthService } from '../../services/auth.service';


@Component({
  selector: 'app-home',
  imports: [],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css',
})
export class HomeComponent {
  nombreUsuario: string | null = null ;
  constructor(private service: AuthService) {}

  ngOnInit() {
   this.service.uvus$.subscribe((uvus) => {
    if (uvus) {
      this.nombreUsuario = uvus.charAt(0).toUpperCase() + uvus?.slice(1).toLocaleLowerCase();
    } else {
      this.nombreUsuario = null;
    }
    });

    console.log('UVUS del usuario:', this.nombreUsuario);
  }
}
