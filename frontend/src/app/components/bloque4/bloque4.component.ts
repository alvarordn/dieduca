import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CircuitosService } from '../../services/circuitos.service';
import Swal from 'sweetalert2';
import { Circuit4ViewComponent } from '../circuit4-view/circuit4-view.component';

@Component({
  selector: 'app-bloque4',
  standalone: true,
  imports: [CommonModule, Circuit4ViewComponent],
  templateUrl: './bloque4.component.html',
  styleUrls: ['./bloque4.component.css']
})
export class Bloque4Component implements OnInit {
  public circuitData: any = null;
  public problemaSeleccionado: number = 1;
  public cargando: boolean = false;

  constructor(private circuitosService: CircuitosService) {}

  ngOnInit(): void {
    this.cargarCircuitoBloque4(this.problemaSeleccionado);
  }

  onCambiarProblema(idProblema: string | number): void {
    this.problemaSeleccionado = Number(idProblema);
    this.cargarCircuitoBloque4(this.problemaSeleccionado);
  }

  cargarCircuitoBloque4(idProblema: number): void {
    this.cargando = true;
    const payload = { bloque: 4, rows: 3, cols: 3, plantilla: idProblema };

    this.circuitosService.generarCircuito(payload).subscribe({
      next: (res: any) => {
        this.cargando = false;
        if (res && res.success && res.circuito) {
          // Asignamos el objeto completo aquí
          this.circuitData = res.circuito;
          console.log('Circuito cargado:', this.circuitData);
        } else {
          Swal.fire('Error', 'Formato de datos incorrecto', 'warning');
        }
      },
      error: (err: any) => {
        this.cargando = false;
        Swal.fire('Error', 'No se pudo conectar con el servidor', 'error');
      }
    });
  }
}