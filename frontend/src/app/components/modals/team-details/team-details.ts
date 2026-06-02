import { Component } from '@angular/core';
import { TeamsService } from '../../../services/teamsService';
import { ActivatedRoute, Route, Router } from '@angular/router';
import { Team } from '../../../models/team';
import { Location } from '@angular/common';

@Component({
  selector: 'app-team-details',
  imports: [],
  templateUrl: './team-details.html',
  styleUrl: './team-details.css',
})
export class TeamDetails {
  equipo!: Team
  constructor(private equipoService: TeamsService,private route: ActivatedRoute,private location: Location){

  }

  ngOnInit(){
    const id = this.route.snapshot.paramMap.get('id')
    this.equipoService.getEquipo(Number(id)).subscribe(data =>{
        this.equipo = data
        console.log(data)
    })
  }

  back(){
    this.location.back()
  }
}
