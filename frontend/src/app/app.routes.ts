import { Routes } from '@angular/router';
import { BloqueComponent } from './components/bloque/bloque.component';
import { TeoriaComponent } from './components/teoria/teoria.component';
import { HomeComponent } from './components/home/home.component';
import { LoginComponent } from './components/login/login.component';
import { RegistroComponent } from './components/registro/registro.component';
import { AuthGuard } from './auth.guard';


export const routes: Routes = [
  { path: 'login', component: LoginComponent },
  { path: 'registro', component: RegistroComponent },
  { path: '', component: HomeComponent },  // Página de inicio
  { path: 'bloque/:id/teoria', component: TeoriaComponent },  // Teoría
  { path: 'bloque/:id/ejercicio', component: BloqueComponent, canActivate: [AuthGuard] }, // Ejercicio
  { path: '**', redirectTo: '' }
];
