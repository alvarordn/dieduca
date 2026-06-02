import { Player } from './player';

export interface Team {

  id: number;

  nombre: string;

  pais: string;

  liga: string;

  estadio: string;

  fundacion: number;

  escudoUrl: string;

  puntos: number;

  golesFavor: number;

  golesContra: number;

  jugadores?: Player[];
}