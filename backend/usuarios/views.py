from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import UserRegisterSerializer, UserLoginSerializer
from django.contrib.auth import get_user_model
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny

User = get_user_model()


class RegisterView(APIView):
    permission_classes = ()

    def post(self, request):
        serializer = UserRegisterSerializer(data = request.data)
        if serializer.is_valid():
            user = serializer.save()

            refresh = RefreshToken.for_user(user)

            return Response({
                "uvus": user.username,
                "token": str(refresh.access_token)
            }, status = status.HTTP_201_CREATED)

        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    permission_classes = () # Permitir acceso sin autenticación

    def post(self, request):
        serializer = UserLoginSerializer(data=request.data)

        if serializer.is_valid():
            user = serializer.validated_data['user']

            # Generar el token JWT al iniciar sesión
            refresh = RefreshToken.for_user(user)

            return Response({
                "uvus": user.username,
                "token": str(refresh.access_token)
            }, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny]) # Permite que Angular mande el "tic" sin complicaciones de permisos por ahora
def track_time(request):
    uvus = request.data.get('uvus')
    try:
        # Usamos 'User' que es el modelo que obtuviste con get_user_model()
        user = User.objects.get(uvus=uvus)
        user.minutos_conectado += 1
        user.save()
        return Response({'status': 'ok'}, status=status.HTTP_200_OK)
    except User.DoesNotExist:
        return Response({'error': 'Usuario no encontrado'}, status=status.HTTP_404_NOT_FOUND)