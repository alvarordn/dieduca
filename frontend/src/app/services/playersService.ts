import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Player } from '../models/player';

@Injectable({
  providedIn: 'root',
})
export class PlayersService {

  private url = 'https://datagol-production.up.railway.app/api/jugadores';

  constructor(private http: HttpClient) { }

  // Obtener todos los jugadores
  getJugadores() {
    return this.http.get<Player[]>(this.url);
  }

  // Obtener un jugador por ID
  getJugador(id: number) {
    return this.http.get<Player>(`${this.url}/${id}`);
  }

  // Crear un nuevo jugador
  createJugador(player: Player) {
    return this.http.post<Player>(this.url, player);
  }

  // Actualizar un jugador existente
  updateJugador(id: number, player: Player) {
    return this.http.put<Player>(`${this.url}/${id}`, player);
  }

  // Borrar un jugador
  deleteJugador(id: number) {
    return this.http.delete(`${this.url}/${id}`);
  }
}