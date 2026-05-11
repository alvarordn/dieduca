import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { CommonModule } from '@angular/common';
import { PdfViewerModule } from 'ng2-pdf-viewer';

@Component({
  selector: 'app-teoria',
  standalone: true,
  imports: [CommonModule, PdfViewerModule],
  templateUrl: './teoria.component.html',
  styleUrls: ['./teoria.component.css'],
})
export class TeoriaComponent implements OnInit {

  // URL del PDF que se va a mostrar en el visor
  pdfUrl!: string;

  // ID del bloque que viene por la ruta (URL)
  bloqueId!: number;

    TEMAS_BLOQUES: { [key: number]: string } = {
    1: 'Conceptos fundamentales y leyes de Kirchhof',
    2: 'Circuitos Resistivos con Generadores Ideales',
    3: 'Fuentes reales y circuitos equivalentes',
    4: 'Técnicas de análisis de circuitos',
    5: 'Componentes dinámicos',
    6: 'Análisis de circuitos de corriente continua en distintos regímenes temporales',
    7: 'Resolución de circuitos de corriente alterna sinusoidal',
    8: 'Potencia y energía en circuitos de corriente alterna sinusoidal',
    9: 'Circuitos trifásicos',
    10: 'Potencia en circuitos trifásicos equilibrados',
  };
  tituloBloque: string = '';
  constructor(private route: ActivatedRoute) {}

  ngOnInit(): void {

    // Nos suscribimos a los parámetros de la ruta
    // para detectar cambios en el id del bloque
    this.route.params.subscribe(({ id }) => {

      // Convertimos el id a número porque viene como string
      this.bloqueId = Number(id);
      this.tituloBloque = this.TEMAS_BLOQUES[this.bloqueId] || 'Bloque';

      // Cargamos el PDF correspondiente al bloque
      this.loadPdf();
    });
  }

  // Función principal que decide qué PDF cargar
  private loadPdf(): void {

    // Calculamos en qué “parte” está el bloque
    const parte = this.getParte(this.bloqueId);

    // Construimos la ruta final del PDF
    this.pdfUrl = this.buildPdfPath(parte, this.bloqueId);

  }

  // Función que decide en qué carpeta está cada bloque
  // (esto es una lógica de organización de archivos)
  private getParte(id: number): string {

    // Bloques 1-6 → Parte 1
    if (id < 7) return 'Parte_1';

    // Bloques 7-8 → Parte 2
    if (id < 9) return 'Parte_2';

    // Bloques 9 en adelante → Parte 3
    return 'Parte_3';
  }

  // Construye la ruta final del PDF según estructura de carpetas
  private buildPdfPath(parte: string, id: number): string {
    return `/assets/teoria/${parte}/T${id}/TC_tema_${id}.pdf`;
  }
}