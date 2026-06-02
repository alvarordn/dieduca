import { Component, OnInit } from '@angular/core';
import { Player } from '../../models/player';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { NgClass, SlicePipe } from '@angular/common';
import { Location } from '@angular/common';
import { PlayersService } from '../../services/playersService';

@Component({
  selector: 'app-rankings',
  standalone: true,
  imports: [NgClass],
  templateUrl: './rankings.html',
  styleUrl: './rankings.css',
})
export class Rankings implements OnInit {

  jugadores: Player[] = [];

  goles: Player[] = [];
  asistencias: Player[] = [];
  valorMercado: Player[] = [];

  constructor(
    private jugadorService: PlayersService,
    private location: Location
  ) {}

  // =========================
  // INIT
  // =========================
  ngOnInit() {
    this.loadJugadores();
  }

  // =========================
  // CARGAR JUGADORES
  // =========================
  loadJugadores() {
    this.jugadorService.getJugadores().subscribe({
      next: data => {
        this.jugadores = data;

        this.calcularGoles();
        this.calcularAsistencias();
        this.calcularValorMercado();
      },
      error: err => console.error('Error cargando jugadores', err)
    });
  }

  // =========================
  // RANKING GOLES
  // =========================
  calcularGoles() {
    this.goles = [...this.jugadores]
      .sort((a, b) => (b.goles ?? 0) - (a.goles ?? 0));
  }

  // =========================
  // RANKING ASISTENCIAS
  // =========================
  calcularAsistencias(){
    this.asistencias = [...this.jugadores]
      .sort((a, b) => (b.asistencias ?? 0) - (a.asistencias ?? 0));
  }

  // =========================
  // RANKING VALOR MERCADO
  // =========================
  calcularValorMercado(){
    this.valorMercado = [...this.jugadores]
      .sort((a, b) => (b.valorMercado ?? 0) - (a.valorMercado ?? 0));
  }

  // =========================
  // BACK BUTTON
  // =========================
  back(): void {
    this.location.back();
  }
}