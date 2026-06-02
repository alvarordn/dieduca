import { Component } from '@angular/core';
import { PlayersService } from '../../services/playersService';
import { Player } from '../../models/player';
import { RouterLink } from '@angular/router';
import Swal from 'sweetalert2';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-players',
  standalone: true,
  imports: [RouterLink,FormsModule],
  templateUrl: './players.html',
  styleUrl: './players.css',
})
export class Players {
  jugadores: Player[] = [];
  searchText: string = '';
  constructor(private jugadorService: PlayersService) {}

  ngOnInit() {
    this.jugadorService.getJugadores().subscribe((data) => {
      this.jugadores = data;
      console.log(data);
    });
  }

  deletePlayer(id: number) {
    Swal.fire({
      title: '¿Eliminar jugador?',
      text: 'No podrás recuperarlo',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonColor: '#dc3545',
      cancelButtonColor: '#6c757d',
      confirmButtonText: 'Sí, eliminar',
      cancelButtonText: 'Cancelar',
    }).then((result) => {
      if (result.isConfirmed) {
        this.jugadorService.deleteJugador(id).subscribe(() => {
          this.jugadores = this.jugadores.filter((p) => p.id !== id);

          Swal.fire({
            title: 'Eliminado',
            text: 'Jugador eliminado correctamente',
            icon: 'success',
            showConfirmButton: false
          });
        });
      }
    });
  }
}
