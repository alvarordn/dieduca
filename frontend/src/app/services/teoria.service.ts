// src/app/services/teoria.service.ts

import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';

export interface IBloqueTeoria {
  titulo: string;
  secciones: Array<{
    titulo: string;
    contenido: string;
    imagen_url: string;
    imagen_caption: string;
  }>;
}

export interface IContenidoTeoria {
  [key: number]: IBloqueTeoria;
}

const CONTENIDO_TEORIA: IContenidoTeoria = {
  1: {
    titulo: 'Bloque 1: Conceptos fundamentales y leyes de Kirchhoff',
    secciones: [
      {
        titulo: 'Carga eléctrica',
        contenido: `
          La Carga eléctrica, $q(t)$, es la propiedad de las partículas elementales que constituyen la materia.
          <ul>
            <li>El electrón ($e^-$) es la carga mínima.</li>
            <li>Unidad: Culombio [C].</li>
          </ul>
        `,
        imagen_url: 'assets/images/coulomb.png',
        imagen_caption: 'Charles-Augustin de Coulomb'
      },
      {
        titulo: 'Leyes de Kirchhoff',
        contenido: 'La suma de corrientes en un nudo es cero (LKC) y la suma de tensiones en una malla es cero (LKT).',
        imagen_url: 'assets/images/kirchhoff.png',
        imagen_caption: 'Gustav Kirchhoff'
      }
    ]
  },
  2: {
    titulo: 'Bloque 2: Elementos de Circuitos',
    secciones: [
      {
        titulo: 'Resistencia Eléctrica',
        contenido: 'La Ley de Ohm establece que $V = I \\cdot R$. La resistencia mide la oposición al flujo de corriente.',
        imagen_url: 'assets/images/ohm.png',
        imagen_caption: 'Georg Simon Ohm'
      },
      {
        titulo: 'Condensadores e Inductancias',
        contenido: 'Elementos que almacenan energía en campos eléctricos ($W = \\frac{1}{2}CV^2$) y magnéticos ($W = \\frac{1}{2}LI^2$).',
        imagen_url: 'assets/images/capacitor.png',
        imagen_caption: 'Simbología de elementos pasivos'
      }
    ]
  },
  3: {
    titulo: 'Bloque 3: Métodos de Análisis',
    secciones: [
      {
        titulo: 'Análisis por Nudos',
        contenido: 'Basado en la aplicación sistemática de la LKC para hallar los potenciales de nudo.',
        imagen_url: 'assets/images/nodos.png',
        imagen_caption: 'Diagrama de nudos'
      },
      {
        titulo: 'Análisis por Mallas',
        contenido: 'Basado en la aplicación de la LKT para hallar las corrientes de malla en circuitos planos.',
        imagen_url: 'assets/images/mallas.png',
        imagen_caption: 'Circuito con dos mallas'
      }
    ]
  },
  4: {
    titulo: 'Bloque 4: Teoremas de Circuitos',
    secciones: [
      {
        titulo: 'Teorema de Thévenin',
        contenido: 'Cualquier red lineal puede sustituirse por una fuente de tensión $V_{th}$ en serie con una $R_{th}$.',
        imagen_url: 'assets/images/thevenin.png',
        imagen_caption: 'Circuito equivalente de Thévenin'
      },
      {
        titulo: 'Superposición',
        contenido: 'La respuesta de un circuito lineal con varias fuentes es la suma de las respuestas individuales.',
        imagen_url: 'assets/images/superposicion.png',
        imagen_caption: 'Análisis de fuentes independientes'
      }
    ]
  },
  5: {
    titulo: 'Bloque 5: Régimen Estacionario Sinusoidal',
    secciones: [
      {
        titulo: 'Concepto de Fasor',
        contenido: 'Representación de señales sinusoidales mediante números complejos para simplificar el análisis.',
        imagen_url: 'assets/images/fasor.png',
        imagen_caption: 'Diagrama fasorial'
      },
      {
        titulo: 'Impedancia Compleja',
        contenido: 'Relación entre tensión y corriente fasorial: $\\mathbf{V} = \\mathbf{I} \\cdot \\mathbf{Z}$.',
        imagen_url: 'assets/images/impedancia.png',
        imagen_caption: 'Triángulo de impedancias'
      }
    ]
  }
};

@Injectable({
  providedIn: 'root'
})
export class TeoriaService {
  getTeoria(bloqueId: number): Observable<IBloqueTeoria | undefined> {
    return of(CONTENIDO_TEORIA[bloqueId]); 
  }
}