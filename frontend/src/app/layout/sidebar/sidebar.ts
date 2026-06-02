import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { EventEmitter, Output } from '@angular/core';

@Component({
  selector: 'app-sidebar',
  imports: [RouterLink, RouterLinkActive],
  templateUrl: './sidebar.html',
  styleUrl: './sidebar.css',
})
export class Sidebar {
  // Creamos el emisor del evento
  @Output() closeSidebar = new EventEmitter<void>();

  // Método que se ejecutará al hacer clic
  onToggle() {
    this.closeSidebar.emit();
    console.log('Sidebar toggled');
  }
}
