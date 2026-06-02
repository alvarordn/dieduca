import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { Location } from '@angular/common';

import { Team } from '../../../models/team';
import { Player } from '../../../models/player';

import { PlayersService } from '../../../services/playersService';
import { TeamsService } from '../../../services/teamsService';

@Component({
  selector: 'app-player-form',
  standalone: true,
  imports: [ReactiveFormsModule],
  templateUrl: './player-modal.html',
  styleUrl: './player-modal.css',
})
export class PlayerModal implements OnInit {

  form: FormGroup;

  players: Player[] = [];
  teams: Team[] = [];

  loading = false;
  errorMessage = '';

  positions = [
    'POR','DFC','LD','LI','MCD','MC','MCO','EI','ED','DC'
  ];

  constructor(
    private fb: FormBuilder,
    private route: ActivatedRoute,
    private router: Router,
    private playerService: PlayersService,
    private teamService: TeamsService,
    private location: Location
  ) {

    this.form = this.fb.group({
      id: [-1],
      nombre: ['', Validators.required],
      apellido1: ['', Validators.required],
      edad: [0, Validators.required],
      estatura: [0, Validators.required],
      peso: [0, Validators.required],
      posicion: ['', Validators.required],
      goles: [0],
      asistencias: [0],
      valorMercado: [0],
      nacionalidad: ['', Validators.required],
      fotoUrl: ['', Validators.required],
      equipo: [null, Validators.required],
    });
  }

  ngOnInit(): void {
    this.loadTeams();
    this.loadPlayerIfEdit();
  }

  // =========================
  // CARGAR EQUIPOS
  // =========================
  loadTeams(): void {
    this.teamService.getEquipos().subscribe(data => {
      this.teams = data;
    });
  }

  // =========================
  // EDIT MODE
  // =========================
  loadPlayerIfEdit(): void {
    const id = this.route.snapshot.paramMap.get('id');

    if (id && id !== '-1') {
      this.playerService.getJugador(Number(id)).subscribe(player => {
        this.form.patchValue({
          ...player,
          equipo: player.equipo?.id ?? null
        });
      });
    }
  }

  // =========================
  // SAVE
  // =========================
  save(): void {

    const value = this.form.value;

    const playerToSend = {
      ...value,
      equipo: { id: value.equipo }
    };

    this.loading = true;

    if (value.id === -1) {

      this.playerService.createJugador(playerToSend).subscribe({
        next: () => {
          this.loading = false;
          this.back();
        },
        error: () => {
          this.loading = false;
          this.errorMessage = 'Error creando jugador';
        }
      });

    } else {

      this.playerService.updateJugador(value.id, playerToSend).subscribe({
        next: () => {
          this.loading = false;
          this.back();
        },
        error: () => {
          this.loading = false;
          this.errorMessage = 'Error actualizando jugador';
        }
      });

    }
  }

  // =========================
  // BACK
  // =========================
  back(): void {
    this.location.back();
  }
}