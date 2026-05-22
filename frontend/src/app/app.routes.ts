import { Routes } from '@angular/router';
import { BloqueComponent } from './components/bloque/bloque.component';
import { TeoriaComponent } from './components/teoria/teoria.component';
import { HomeComponent } from './components/home/home.component';
import { LoginComponent } from './components/login/login.component';
import { RegistroComponent } from './components/registro/registro.component';
import { AuthGuard } from './auth.guard';
import { ResultadosComponent } from './components/resultados/resultados.component';
import { Bloque4Component } from './components/bloque4/bloque4.component';

export const routes: Routes = [
  { path: 'login', component: LoginComponent }, // Login
  { path: 'registro', component: RegistroComponent }, // Registro
  { path: '', component: HomeComponent }, // Página de inicio
  {
    path: 'bloque/:id/teoria',
    component: TeoriaComponent,
    canActivate: [AuthGuard],
  }, // Teoría
  {
    path: 'bloque/:id/ejercicio',
    component: BloqueComponent,
    canActivate: [AuthGuard],
  }, // Ejercicio
  {
    path: 'resultados/:id',
    component: ResultadosComponent,
    canActivate: [AuthGuard],
  },
  { path: '**', redirectTo: '' },
  { path: 'bloque/4/ejercicio', component: Bloque4Component },
];
