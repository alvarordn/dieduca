import { Component } from '@angular/core';
import { PlayersService } from '../../../services/playersService';
import { ActivatedRoute } from '@angular/router';
import { Player } from '../../../models/player';
import { Location } from '@angular/common';

@Component({
  selector: 'app-player-details',
  imports: [],
  templateUrl: './player-details.html',
  styleUrl: './player-details.css',
})
export class PlayerDetails {
  jugador!: Player
  constructor(private jugadorService: PlayersService,private route: ActivatedRoute,private location: Location){

  }

  ngOnInit(){
    const id = this.route.snapshot.paramMap.get('id')
    this.jugadorService.getJugador(Number(id)).subscribe(data =>{
        this.jugador = data
    })
  }

  back(){
    this.location.back()
  }
}
