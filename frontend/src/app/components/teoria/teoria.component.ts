import { Component } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { ActivatedRoute } from '@angular/router';
import { CommonModule } from '@angular/common';

import { PdfViewerModule } from 'ng2-pdf-viewer';

@Component({
  selector: 'app-teoria',
  standalone: true,
  imports: [PdfViewerModule,CommonModule],
  templateUrl: './teoria.component.html',
  styleUrls: ['./teoria.component.css'],
})
export class TeoriaComponent {
  pdfSafeUrl!: SafeResourceUrl;
  bloqueId = 0;

  constructor(
    private sanitizer: DomSanitizer,
    private route: ActivatedRoute,
  ) {}

  ngOnInit() {
    this.route.params.subscribe((params) => {
      //obtenemos el bloque id
      this.bloqueId = +params['id'];
      
      //dividimos en partes para poder cambiar de temas 
      let parte = '';
      if (this.bloqueId < 7) {
        parte = 'Parte_1';
      } else if (this.bloqueId < 9) {
        parte = 'Parte_2';
      } else {
        parte = 'Parte_3';
      }

      //asignamos la ruta con las parte y el bloqueId para saber q tema es
      const rutaPdf = `assets/teoria/${parte}/T${this.bloqueId}/TC_tema_${this.bloqueId}.pdf`;
      this.pdfSafeUrl = this.sanitizer.bypassSecurityTrustResourceUrl(rutaPdf);
    });
    
  }
}
