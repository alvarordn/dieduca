export interface Componente {
  id: string;
  source: string;
  target: string;
  type: string;
  value: string;
  orientation: 'horizontal' | 'vertical';
  labelPosition: string;
}