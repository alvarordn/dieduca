from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes
from .alvaro.lib_new import Circuit

# api_view: seria un decorador el cual se coloca encima de cada funcion, y le dice a la funcion que será una API.
# En Django normal, al crear una vista, se devuelve una pagina HTML para que se vea en el navegador, pero en este caso
# si queremos crear una API, no se envia HTML, si no datos como datos JSON, que otras aplicaciones puedan leer, como en este caso Angular.

# Response: es para devolver los datos en formato JSON.
# Status: es para los codigos HTTP.
# base64: para convertir las imagenes resultantes en texto enviable por JSON.
# io: para trabajar en memoria y no con archivos en disco.
# matplotlib.use('Agg'): matplotlib abre ventanas graficas, en un servidor no hay pantalla, y lo que hace 'Agg', es decirle que genere la
# imagen en memoria, sin ventanas.

@api_view(['GET']) # Solo responde a peticiones GET, como cuando se pone una URL.
def test_connection(request):
    return Response({"mensaje": "Conexión establecida correctamente"}, status=status.HTTP_200_OK)

# Devuelve un simple mensaje en JSON.

def determinar_tipo_nodo(row, col, total_rows, total_cols):
    es_primera_fila = (row == 0)
    es_ultima_fila = (row == total_rows - 1)
    es_primera_col = (col == 0)
    es_ultima_col = (col == total_cols - 1)

    if es_primera_fila and es_primera_col: return 'corner-top-left'
    if es_primera_fila and es_ultima_col: return 'corner-top-right'
    if es_ultima_fila and es_primera_col: return 'corner-bottom-left'
    if es_ultima_fila and es_ultima_col: return 'corner-bottom-right'

    if es_primera_fila: return 'edge-top'
    if es_ultima_fila: return 'edge-bottom'
    if es_primera_col: return 'edge-left'
    if es_ultima_col: return 'edge-right'

    return 'center'

# Esto determina exactamente que tipo de nodo generar, las esquinas, o el centro si es que el circuito tiene.


def calcular_posicion_etiqueta(source_row, source_col, target_row, target_col, total_rows, total_cols, orientation):
    mid_row = (source_row + target_row) / 2.0
    mid_col = (source_col + target_col) / 2.0

    if mid_row < 0.5: return 'outside-top'
    if mid_row > total_rows - 1.5: return 'outside-bottom'
    if mid_col < 0.5: return 'outside-left'
    if mid_col > total_cols - 1.5: return 'outside-right'

    return 'inside-bottom' if orientation == 'horizontal' else 'inside-right'

# Aqui esta la Api a la que se accederá

@api_view(['POST', 'GET'])
@permission_classes([AllowAny])
def generar_circuito(request):
    try:
        if request.method == 'GET':
            bloque_id, rows, cols = '1', 2, 3
        else:
            bloque_id = request.data.get('bloque', '1')
            rows = int(request.data.get('rows', 2))
            cols = int(request.data.get('cols', 3))

        circuito_motor = Circuit(rows=rows, cols=cols)

        intentos = 0
        while not circuito_motor.solve() and intentos < 10:
            circuito_motor = Circuit(rows=rows, cols=cols)
            intentos += 1

        nodos = []
        for node in circuito_motor.G.nodes():
            idx = int(node[1])
            jdx = int(node[2])

            node_type = determinar_tipo_nodo(idx, jdx, rows, cols)

            nodos.append({
                'id': node,
                'row': idx,
                'col': jdx,
                'type': node_type,
                'potential': str(circuito_motor.G.nodes[node].get('potential', 0))
            })

        componentes = []
        for comp_id, (source, target) in enumerate(circuito_motor.G.edges()):
            source_row, source_col = int(source[1]), int(source[2])
            target_row, target_col = int(target[1]), int(target[2])

            orientation = 'horizontal' if source_row == target_row else 'vertical'

            # Esto determina si los nodos van:
            # Seguidos (vertical)
            # Uno abajo del otro (horizontal)
            label_position = calcular_posicion_etiqueta(source_row, source_col, target_row, target_col, rows, cols, orientation)

            edge_data = circuito_motor.G[source][target]
            componentes.append({
                'id': f'comp-{comp_id}',
                'source': source,
                'target': target,
                'type': edge_data.get('element'),
                'value': edge_data.get('string'),
                'orientation': orientation,
                'labelPosition': label_position
            })

        return Response({
            'success': True,
            'mensaje': 'Circuito generado correctamente',
            'circuito': {
                'rows': rows,
                'cols': cols,
                'nodos': nodos,
                'componentes': componentes
            }
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)


