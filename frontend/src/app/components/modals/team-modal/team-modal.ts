import { Component } from '@angular/core';
import { CommonModule, Location } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import {
  FormBuilder,
  FormGroup,
  ReactiveFormsModule,
  Validators
} from '@angular/forms';

import { TeamsService } from '../../../services/teamsService';
import { Team } from '../../../models/team';

@Component({
  selector: 'app-team-modal',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './team-modal.html',
  styleUrl: './team-modal.css',
})
export class TeamModal {

  form: FormGroup;

  errorMessage = '';

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private teamService: TeamsService,
    private location: Location,
    private fb: FormBuilder
  ) {

    this.form = this.fb.group({
      id: [-1],

      nombre: ['', Validators.required],

      pais: ['', Validators.required],

      liga: ['', Validators.required],

      estadio: ['', Validators.required],

      fundacion: [0, Validators.required],

      escudoUrl: ['', Validators.required],

      puntos: [0],

      golesFavor: [0],

      golesContra: [0],

      jugadores: [[]]
    });
  }

  // =========================
  // INIT
  // =========================
  ngOnInit(): void {

    const idTeam = this.route.snapshot.paramMap.get('id');

    // EDIT MODE
    if (idTeam && idTeam !== '-1') {

      this.teamService.getEquipo(Number(idTeam)).subscribe((data: Team) => {

        this.form.patchValue({
          ...data,
          jugadores: data.jugadores || []
        });

      });

    }
  }

  // =========================
  // SAVE
  // =========================
  save(): void {

    if (!this.form.valid) {

      this.errorMessage =
        'Por favor completa todos los campos obligatorios';

      return;
    }

    const value = this.form.value;

    const { jugadores, ...teamToSend } = value;

    const idTeam = this.route.snapshot.paramMap.get('id');

    // CREATE
    if (idTeam === '-1' || idTeam === null) {

      this.teamService.createEquipo(teamToSend).subscribe({

        next: () => {
          this.router.navigate(['/teams']);
        },

        error: () => {
          this.errorMessage = 'Error creando equipo';
        }

      });

    }

    // UPDATE
    else {

      this.teamService
        .updateEquipo(Number(idTeam), teamToSend)
        .subscribe({

          next: () => {
            this.router.navigate(['/teams']);
          },

          error: () => {
            this.errorMessage = 'Error actualizando equipo';
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