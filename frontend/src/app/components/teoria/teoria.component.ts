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

  constructor(private route: ActivatedRoute) {}

  ngOnInit(): void {

    // Nos suscribimos a los parámetros de la ruta
    // para detectar cambios en el id del bloque
    this.route.params.subscribe(({ id }) => {

      // Convertimos el id a número porque viene como string
      this.bloqueId = Number(id);

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

    // Debug para ver si la ruta está bien generada
    console.log('PDF URL:', this.pdfUrl);
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