import { Grado } from './grado';

export interface Usuario {
    id?: number;      
    uvus: string;    
    email: string;    
    grado: string | Grado; 
    password: string; 
}
