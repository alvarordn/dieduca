import { HttpClient } from '@angular/common/http';
import { Component, Input } from '@angular/core';
import { ActivatedRoute, Route, RouterLink } from '@angular/router';

@Component({
  selector: 'app-resultados',
  imports: [],
  templateUrl: './resultados.component.html',
  styleUrl: './resultados.component.css'
})
export class ResultadosComponent {
  public fallos = 0;
  public aciertos = 0;
  public exito = 0;
  public idUsuario = 0;
  public totalPreguntas = 0;
  public nombreUsuario: string = "";

  constructor(private http: HttpClient, private route: ActivatedRoute){
    
  }


  ngOnInit(){
    this.idUsuario = this.route.snapshot.params["id"]
    console.log(this.idUsuario);
    this.nombreUsuario = localStorage.getItem("uvus")?.toLocaleUpperCase() || ""
  }
}
