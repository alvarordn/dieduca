import { Component } from '@angular/core';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-home',
  imports: [],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent {

  nombreUsuario: string | null = ''; 
  constructor(private service: AuthService) {}

  ngOnInit() {
    this.nombreUsuario = this.service.getUvus();
    console.log('UVUS del usuario:', this.nombreUsuario);
  }
}
