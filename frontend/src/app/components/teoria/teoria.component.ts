import { Component } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-teoria',
  templateUrl: './teoria.component.html',
  styleUrls: ['./teoria.component.css'],
})
export class TeoriaComponent {
  pdfSafeUrl!: SafeResourceUrl;
  bloqueId = 0;

  constructor(
    private sanitizer: DomSanitizer,
    private route: ActivatedRoute,
  ) {

  }

  ngOnInit() {
    this.route.params.subscribe((params) => {
      this.bloqueId = +params['id'];
      let parte = '';
      if (this.bloqueId < 7) {
        parte = 'Parte_1';
      } else if (this.bloqueId < 9) {
        parte = 'Parte_2';
      } else {
        parte = 'Parte_3';
      }
      const rutaPdf = `assets/teoria/${parte}/T${this.bloqueId}/TC_tema_${this.bloqueId}.pdf`;
      this.pdfSafeUrl = this.sanitizer.bypassSecurityTrustResourceUrl(rutaPdf);
    });
  }
}
