import { Grado } from './grado';

export interface Usuario {
    id?: number;      // ID único generado por la base de datos 
    uvus: string;     // El identificador de la universidad 
    email: string;    // Correo único 
    grado: string | Grado; // Puede ser el string 'GITI' o el objeto completo
    password?: string; // Opcional, solo se usa en el registro 
}
