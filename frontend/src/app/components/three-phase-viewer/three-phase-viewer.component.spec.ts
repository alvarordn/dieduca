import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ThreePhaseViewerComponent } from './three-phase-viewer.component';

describe('ThreePhaseViewerComponent', () => {
  let component: ThreePhaseViewerComponent;
  let fixture: ComponentFixture<ThreePhaseViewerComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ThreePhaseViewerComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ThreePhaseViewerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
