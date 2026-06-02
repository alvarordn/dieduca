import { Component } from '@angular/core';
import { TeamsService } from '../../services/teamsService';
import { Team } from '../../models/team';
import { NgClass } from '@angular/common';

@Component({
  selector: 'app-standings',
  imports: [NgClass],
  templateUrl: './standings.html',
  styleUrl: './standings.css',
})
export class Standings {
  equipos: Team[] = [];
    constructor(private equipoService: TeamsService){
  
    }
  
    ngOnInit(){
      this.equipoService.getEquipos().subscribe( data => {
        this.equipos = data.sort((a, b) => b.puntos - a.puntos);
      })
    }
}
