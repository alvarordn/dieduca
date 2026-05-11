import { Component, OnInit, OnDestroy, HostListener } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { NavbarComponent } from './components/navbar/navbar.component';
import { FooterComponent } from './components/footer/footer.component';
import { HttpClient, HttpClientModule } from '@angular/common/http';

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

  private estarActivo = true;

  private readonly tiempoInactivo = 120000;

  constructor(private http: HttpClient) {}

  ngOnInit() {

    this.startTracking();
    this.resetInactividad();
  }

  @HostListener("window:mousemove")
  @HostListener('window:keydown')
  @HostListener('window:click')
  @HostListener('window:scroll')

  onUserActivity() {

    this.estarActivo = true;
    this.resetInactividad();
  }

  resetInactividad() {

    clearTimeout(this.timeOutId);

    this.timeOutId = setTimeout(() => {

      this.estarActivo = false;



    }, this.tiempoInactivo);
  }

  startTracking() {

    this.intervalId = setInterval(() => {

      const userUvus = sessionStorage.getItem('uvus');

      // condición directa sin comparaciones innecesarias
      if (userUvus && this.estarActivo) {

        this.http.post(
          'http://localhost:8000/api/auth/track-time/',
          { uvus: userUvus }
        ).subscribe({
        });
      }

    }, 60000);
  }

  ngOnDestroy() {

    if (this.intervalId) {
      clearInterval(this.intervalId);
    }

    if (this.timeOutId) {
      clearTimeout(this.timeOutId);
    }
  }
}