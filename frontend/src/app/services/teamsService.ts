import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Team } from '../models/team';

@Injectable({
  providedIn: 'root',
})
export class TeamsService {

  private url = 'https://datagol-production.up.railway.app/api/equipos';

  constructor(private http: HttpClient) { }

  getEquipos() {
    return this.http.get<Team[]>(this.url);
  }

  getEquipo(id: number) {
    return this.http.get<Team>(`${this.url}/${id}`);
  }

  createEquipo(team: Team) {
    return this.http.post<Team>(this.url, team);
  }

  updateEquipo(id: number, team: Team) {
    return this.http.put<Team>(`${this.url}/${id}`, team);
  }

  deleteEquipo(id: number) {
    return this.http.delete(`${this.url}/${id}`);
  }
}