import { Component, OnInit, OnDestroy } from '@angular/core'; // Añadimos OnInit y OnDestroy
import { RouterOutlet } from '@angular/router';
import { NavbarComponent } from './components/navbar/navbar.component';
import { FooterComponent } from './components/footer/footer.component';
import { HttpClient, HttpClientModule } from '@angular/common/http'; // Necesario para la API

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, NavbarComponent, FooterComponent, HttpClientModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'frontend';
  private intervalId: any;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.startTracking();
  }

  startTracking() {
    // Ejecutamos el envío cada 60 segundos (1 minuto)
    this.intervalId = setInterval(() => {
      // Obtenemos el uvus del localStorage (asegúrate de guardarlo ahí al hacer login)
      const userUvus = localStorage.getItem('uvus'); 

      if (userUvus) {
        this.http.post('http://localhost:8000/api/auth/track-time/', { uvus: userUvus })
          .subscribe({
            next: () => console.log('Minuto de conexión registrado'),
            error: (err) => console.error('Error al registrar tiempo', err)
          });
      }
    }, 60000); 
  }

  ngOnDestroy() {
    // Limpiamos el contador si se destruye el componente para evitar fugas de memoria
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
  }
}