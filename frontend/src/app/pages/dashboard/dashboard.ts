import { Component, OnInit } from '@angular/core';
import { TeamsService } from '../../services/teamsService';
import { Team } from '../../models/team';
import { Player } from '../../models/player';
import { PlayersService } from '../../services/playersService';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.html',
  styleUrls: ['./dashboard.css']
})
export class Dashboard implements OnInit {

  equipos: Team[] = [];
  players: Player[] = [];

  loading = true;

  recentActivity: string[] = [];

  constructor(
    private teamsService: TeamsService,
    private jugadorService: PlayersService
  ) {}

  ngOnInit(): void {
    this.loadData();
  }

  loadData() {
    this.loading = true;

    this.teamsService.getEquipos().subscribe(teams => {
      this.equipos = teams;
      console.log(this.equipos)

      this.jugadorService.getJugadores().subscribe(players => {
        this.players = players;
        this.loading = false;
      });
    });
  }
}