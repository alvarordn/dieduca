import { Component } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-teoria',
  templateUrl: './teoria.component.html',
  styleUrls: ['./teoria.component.css']
})
export class TeoriaComponent {
  pdfSafeUrl: SafeResourceUrl;
  bloqueId = 0;

  constructor(private sanitizer: DomSanitizer, private route: ActivatedRoute) {
    // Convertir el PDF a URL segura para Angular
    this.pdfSafeUrl = this.sanitizer.bypassSecurityTrustResourceUrl('assets/teoria/Prueba.pdf');
  }

  ngOnInit() {
    this.route.params.subscribe(params => {
      this.bloqueId = +params['id'];
    });
  }
}