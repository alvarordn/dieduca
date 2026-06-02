import { Component, OnInit } from '@angular/core';
import { TeamsService } from '../../services/teamsService';
import { Team } from '../../models/team';
import { RouterLink } from '@angular/router';
import Swal from 'sweetalert2';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-teams',
  templateUrl: './teams.html',
  standalone: true,
  styleUrls: ['./teams.css'],
  imports: [RouterLink, FormsModule],
})
export class Teams {
  searchText: string = '';
  equipos: Team[] = [];
  constructor(private equipoService: TeamsService) {}

  ngOnInit() {
    this.equipoService.getEquipos().subscribe((data) => {
      this.equipos = data;
    });
  }

  deleteTeam(id: number) {
    Swal.fire({
      title: '¿Eliminar equipo?',
      text: 'No podrás recuperar este equipo',
      icon: 'warning',

      showCancelButton: true,

      confirmButtonColor: '#dc3545',
      cancelButtonColor: '#6c757d',

      confirmButtonText: 'Sí, eliminar',
      cancelButtonText: 'Cancelar',
    }).then((result) => {
      if (result.isConfirmed) {
        this.equipoService.deleteEquipo(id).subscribe({
          next: () => {
            // quitar del array
            this.equipos = this.equipos.filter((team) => team.id !== id);

            Swal.fire({
              title: 'Eliminado',
              text: 'Equipo eliminado correctamente',
              icon: 'success',
              timer: 1500,
              showConfirmButton: false,
            });
          },

          error: () => {
            Swal.fire({
              title: 'Error',
              text: 'No se pudo eliminar el equipo',
              icon: 'error',
            });
          },
        });
      }
    });
  }
}
