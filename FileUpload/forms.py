from django import forms
from .models import csv_path
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import get_user_model

User = get_user_model()
class UploadCSV(forms.ModelForm):
     class Meta:
        model = csv_path
        fields = ['csv_file']
        
class SignUp(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def clean_email(self):
        email = self.cleaned_data.get('email') #the clean email is used to get the already enterd email data i.e after the validation 
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email already in use")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user
    
class Login(forms.Form):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)

    def clean(self):
        cleaned_data = super().clean()
        email = cleaned_data.get('email')
        password = cleaned_data.get('password')

        if not email or not password:
            raise forms.ValidationError("Email and password are required.")

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            raise forms.ValidationError("Invalid email or password")

        if not user.check_password(password):
            raise forms.ValidationError("Invalid email or password")

        self.user = user  # Store the user object for the view to use
        return cleaned_data


