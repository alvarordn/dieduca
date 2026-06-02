import { Team } from './team';

export interface Player {
  id: number;

  // IDENTIDAD
  nombre: string;
  apellido1: string;
  apellido2: string;

  // FÍSICO
  edad: number;
  estatura: number;
  peso: number;

  // FÚTBOL
  posicion: string;
  goles: number;
  asistencias: number;

  // VALOR
  valorMercado: number;

  // EXTRAS
  nacionalidad: string;

  // FOTO
  fotoUrl: string;

  // RELACIÓN
  equipo: Team | null;
}
