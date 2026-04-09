import { Component, OnInit, OnDestroy, HostListener } from '@angular/core'; // Añadimos OnInit y OnDestroy
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
  private timeOutId: any;
  private estarActivo: boolean = true;
  private readonly tiempoInactivo = 120000; //2 minutos de inactividad

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.startTracking();
    this.resetInactividad()
  }
  @HostListener("window:mousemove")
  @HostListener('window:keydown')
  @HostListener('window:click')
  @HostListener('window:scroll')
  onUserActivity(){
    this.estarActivo = true;
    this.resetInactividad()
  }

  resetInactividad(){
    clearTimeout(this.timeOutId)
    //si pasan 2 minutos inactivo se ejecuta esto
    this.timeOutId = setTimeout(()=>{
      this.estarActivo = false;
      console.log("Usuario Inactivo")
    },this.tiempoInactivo)
  }

  startTracking() {
    if(this.estarActivo){
      console.log("usuario activo")
    }
    // Ejecutamos el envío cada 60 segundos (1 minuto)
    this.intervalId = setInterval(() => {
      // Obtenemos el uvus del sessionStorage
      const userUvus = sessionStorage.getItem('uvus'); 
      //si tenemos uvus y esta activo se ejecuta esto 
      if (userUvus && this.estarActivo) {
        this.http.post('http://localhost:8000/api/auth/track-time/', { uvus: userUvus })
          .subscribe({
            next: () => console.log('Minuto de conexión registrado' ),
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
    if (this.timeOutId) clearTimeout(this.timeOutId)
  }
}