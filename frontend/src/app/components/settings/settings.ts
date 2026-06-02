import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-settings',
  templateUrl: './settings.html',
  styleUrl: './settings.css',
})
export class Settings implements OnInit {

  ngOnInit(): void {
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark';

    if (savedTheme) {
      document.body.classList.remove('light', 'dark');
      document.body.classList.add(savedTheme);
    } else {
      document.body.classList.add('dark');
    }
  }

  setTheme(theme: 'light' | 'dark'): void {
    document.body.classList.remove('light', 'dark');
    document.body.classList.add(theme);

    localStorage.setItem('theme', theme);
  }
}