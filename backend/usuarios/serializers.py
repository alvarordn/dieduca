from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth import authenticate

User = get_user_model()

# 1. Serializador de Registro
class UserRegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ('uvus', 'email', 'grado', 'password')

    def create(self, validated_data):
        uvus_valor = validated_data.get('uvus')
        user = User.objects.create_user(
            username=uvus_valor,
            uvus=uvus_valor,
            email=validated_data['email'],
            password=validated_data['password'],
            grado=validated_data.get('grado', '')
        )
        return user

# 2. Serializador de Login
class UserLoginSerializer(serializers.Serializer):
    uvus = serializers.CharField(required=True)
    password = serializers.CharField(write_only=True, required=True)

    def validate(self, data):
        uvus = data.get("uvus")
        password = data.get("password")
        user = authenticate(username=uvus, password=password)

        if user is None:
            raise serializers.ValidationError("UVUS o contraseña incorrectos.")

        data['user'] = user
        return data