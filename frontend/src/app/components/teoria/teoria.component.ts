import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { MathJaxDirective } from '../../math-jax.directive';
import { TeoriaService } from '../../services/teoria.service';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

@Component({
  selector: 'app-teoria',
  standalone: true,
  imports: [CommonModule, MathJaxDirective],
  templateUrl: './teoria.component.html',
  styleUrl: './teoria.component.css'
})
export class TeoriaComponent implements OnInit {
  bloqueId: number = 0;
  teoria: any = null;
  
  // 1. Variable para el [innerHTML] (Sanitizada para Angular)
  contenidoTeoriaRenderizado: SafeHtml = ''; 
  // 2. Variable para la directiva [appMathJax] (String puro para procesar $)
  contenidoParaMathJax: string = '';

  constructor(
    private route: ActivatedRoute,
    private teoriaService: TeoriaService,
    private sanitizer: DomSanitizer
  ) {}

  ngOnInit() {
    this.route.params.subscribe(params => {
      this.bloqueId = +params['id'];
      this.cargarDatos();
    });
  }

  cargarDatos() {
    // Cargar metadatos del JSON (título, secciones extra, etc.)
    this.teoriaService.getTeoria(this.bloqueId).subscribe(data => {
      this.teoria = data;
    });

    // Cargar el archivo .tex con la teoría de circuitos
    this.teoriaService.getArchivoLatex("Prueba.tex").subscribe({
      next: (data) => {
        const htmlLimpio = this.procesarBeamer(data);
        
        // Sincronizamos: MathJax necesita el texto, la vista necesita el SafeHtml
        this.contenidoParaMathJax = htmlLimpio;
        this.contenidoTeoriaRenderizado = this.sanitizer.bypassSecurityTrustHtml(htmlLimpio);
      },
      error: (err) => console.error("Error cargando el archivo LaTeX", err)
    });
  }

  procesarBeamer(latex: string): string {
    if (!latex) return '';

    let res = latex;

    // A. LIMPIEZA INICIAL: Comentarios (%) y saltos de línea de LaTeX (\\)
    res = res.replace(/%.*?\n/g, ''); // Elimina comentarios de autor
    res = res.replace(/\\\\/g, '<br>'); // Convierte saltos de línea de LaTeX en HTML
    res = res.replace(/\ufffd/g, ''); // Limpieza de rombos residuales

    // B. EXTRAER EL CUERPO DEL DOCUMENTO
    const start = res.indexOf('\\begin{document}');
    const end = res.lastIndexOf('\\end{document}');
    if (start !== -1 && end !== -1) {
      res = res.substring(start + 16, end);
    }

    // C. ARREGLO DE COMANDOS ESPECIALES (Coulomb, Center, etc.)
    res = res
      // Convierte \tiny{Texto} en un pie de foto centrado y pequeño
      .replace(/\\tiny\s*\{([\s\S]*?)\}/g, '<div class="tiny-caption">$1</div>')
      // Maneja entornos de centrado
      .replace(/\\begin{center}/g, '<div class="latex-center">')
      .replace(/\\end{center}/g, '</div>')
      .replace(/Unknown environment 'center'/g, '')
      // Elimina figuras y columnas de Beamer (incompatibles con web)
      .replace(/\\begin{figure}[\s\S]*?\\end{figure}/g, '') 
      .replace(/\\begin{columns}[\s\S]*?\\end{columns}/g, (match) => {
          return match.replace(/\\begin{column}.*?\}|\\end{column}|\\begin{columns}|\\end{columns}/g, '');
      });

    // D. TRANSFORMACIÓN A ESTRUCTURA WEB (Basado en tu CSS de Jakarta y Lora)
    res = res
      .replace(/\\section{(.*?)}/g, '<h2 class="section-divider">$1</h2>')
      .replace(/\\begin{frame}/g, '<div class="slide-card">')
      .replace(/\\end{frame}/g, '</div>')
      .replace(/\\frametitle{(.*?)}/g, '<h3 class="slide-title">$1</h3>')
      .replace(/\\textbf{(.*?)}/g, '<strong>$1</strong>')
      .replace(/\\structure{(.*?)}/g, '<span class="text-accent">$1</span>')
      .replace(/\\begin{itemize}/g, '<ul>')
      .replace(/\\end{itemize}/g, '</ul>')
      .replace(/\\item/g, '<li>')
      .replace(/\\vspace{.*?}|\\medskip|\\titlepage|\\label{.*?}/g, '');

    return res;
  }
}