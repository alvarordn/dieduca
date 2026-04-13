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

  pdfUrl!: string;
  bloqueId!: number;

  constructor(private route: ActivatedRoute) {}

  ngOnInit(): void {
    this.route.params.subscribe(({ id }) => {
      this.bloqueId = Number(id);
      this.loadPdf();
    });
  }

  private loadPdf(): void {
    const parte = this.getParte(this.bloqueId);
    this.pdfUrl = this.buildPdfPath(parte, this.bloqueId);

    console.log('PDF URL:', this.pdfUrl);
  }

  private getParte(id: number): string {
    if (id < 7) return 'Parte_1';
    if (id < 9) return 'Parte_2';
    return 'Parte_3';
  }

  private buildPdfPath(parte: string, id: number): string {
    return `/assets/teoria/${parte}/T${id}/TC_tema_${id}.pdf`;
  }
}