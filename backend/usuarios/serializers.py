from django.shortcuts import render
from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth import authenticate


# Create your views here.

User = get_user_model() # Obtiene el modelo de usuario activo

class UserRegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, style={'input_type': 'password'})
    
    class Meta:
        model = User
        fields = ('uvus', 'email', 'grado', 'password')
        
    def create(self, validated_data):
        
        grado = validated_data.pop('grado', None)
        uvus_valor = validated_data.get('uvus')
        
        user = User.objects.create_user(
            username=uvus_valor,
            email=validated_data['email'],
            password=validated_data['password']
        )
        user.uvus = uvus_valor
        user.grado = grado
        user.save()
        
        return user
    
class UserLoginSerializer(serializers.Serializer):
    uvus = serializers.CharField(required=True)
    password = serializers.CharField(write_only=True, required=True)
    
    def validate(self, data):
        uvus = data.get("uvus")
        password = data.get("password")
        
        user = authenticate(username=uvus, password=password)
        
        if user is None:
            raise serializers.ValidationError("UVUS o contraseña incorrectos.")
        
        if not user.is_active:
            raise serializers.ValidationError("Usuario inactivo.")
        
        data['user'] = user
        return data