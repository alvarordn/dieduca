// src/app/services/teoria.service.ts

import { HttpClient } from '@angular/common/http';
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
        imagen_caption: 'Charles-Augustin de Coulomb',
      },
      {
        titulo: 'Leyes de Kirchhoff',
        contenido:
          'La suma de corrientes en un nudo es cero (LKC) y la suma de tensiones en una malla es cero (LKT).',
        imagen_url: 'assets/images/kirchhoff.jpg',
        imagen_caption: 'Gustav Kirchhoff',
      },
    ],
  },
  2: {
    titulo: 'Bloque 2: Elementos de Circuitos',
    secciones: [
      {
        titulo: 'Resistencia Eléctrica (Georg Simon Ohm)',
        contenido:
          'La Ley de Ohm establece que $V = I \\cdot R$. Ohm descubrió que la corriente que fluye por un conductor es proporcional a la diferencia de potencial.',
        imagen_url: 'assets/images/Ohm.webp',
        imagen_caption: 'Georg Simon Ohm (1789-1854)',
      },
      {
        titulo: 'Condensadores e Inductancias (Michael Faraday)',
        contenido:
          'Elementos que almacenan energía en campos eléctricos ($W = \\frac{1}{2}CV^2$) y magnéticos ($W = \\frac{1}{2}LI^2$). Faraday sentó las bases de la inducción electromagnética.',
        imagen_url: 'assets/images/Michael-faraday.jpg',
        imagen_caption: 'Michael Faraday (1791-1867)',
      },
    ],
  },
  3: {
    titulo: 'Bloque 3: Métodos de Análisis',
    secciones: [
      {
        titulo: 'Análisis por Nudos (James Clerk Maxwell)',
        contenido:
          'Basado en la aplicación sistemática de la LKC. Maxwell unificó el análisis de redes permitiendo resolver circuitos de gran complejidad.',
        imagen_url: 'assets/images/James-Clerk-Maxwell.jpg',
        imagen_caption: 'James Clerk Maxwell (1831-1879)',
      },
      {
        titulo: 'Análisis por Mallas (André-Marie Ampère)',
        contenido:
          'Basado en la aplicación de la LKT. El análisis de mallas utiliza la corriente como variable fundamental, nombrada Amperio en su honor.',
        imagen_url: 'assets/images/andre-marie.jpg',
        imagen_caption: 'André-Marie Ampère (1775-1836)',
      },
    ],
  },
  4: {
    titulo: 'Bloque 4: Teoremas de Circuitos',
    secciones: [
      {
        titulo: 'Teorema de Thévenin (Léon Charles Thévenin)',
        contenido:
          'Cualquier red lineal puede sustituirse por una fuente de tensión $V_{th}$ en serie con una $R_{th}$. Thévenin fue un ingeniero clave en la telegrafía francesa.',
        imagen_url: 'assets/images/leon-charles-thevenin.jpg',
        imagen_caption: 'Léon Charles Thévenin (1857-1926)',
      },
      {
        titulo: 'Superposición (Edward Lawry Norton)',
        contenido:
          'La respuesta de un circuito lineal con varias fuentes es la suma de las individuales. Norton extendió estos conceptos para crear equivalentes de corriente.',
        imagen_url: 'assets/images/Edward_Lawry_Norton.jpg',
        imagen_caption: 'Edward Lawry Norton (1898-1983)',
      },
    ],
  },
  5: {
    titulo: 'Bloque 5: Régimen Estacionario Sinusoidal',
    secciones: [
      {
        titulo: 'Concepto de Fasor (Charles Proteus Steinmetz)',
        contenido:
          'Representación mediante números complejos. Steinmetz permitió que la ingeniería eléctrica pasara de cálculos infernales a aritmética compleja sencilla.',
        imagen_url: 'assets/images/charles-proteus.webp',
        imagen_caption: 'Charles Proteus Steinmetz (1865-1923)',
      },
      {
        titulo: 'Impedancia Compleja (Nikola Tesla)',
        contenido:
          'Relación $\\mathbf{V} = \\mathbf{I} \\cdot \\mathbf{Z}$. Tesla fue el mayor promotor del sistema de corriente alterna que analizamos fasorialmente.',
        imagen_url: 'assets/images/nikola-tesla.jpeg',
        imagen_caption: 'Nikola Tesla (1856-1943)',
      },
    ],
  },
};

@Injectable({
  providedIn: 'root',
})
export class TeoriaService {
  constructor(private http: HttpClient) {}

  getArchivoLatex(nombre: String) {
    return this.http.get(`assets/teoria/${nombre}`, {responseType: "text"})
  }
  getTeoria(bloqueId: number): Observable<IBloqueTeoria | undefined> {
    return of(CONTENIDO_TEORIA[bloqueId]);
  }
}
