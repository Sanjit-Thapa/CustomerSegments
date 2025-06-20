from django import forms
from .models import csv_path
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import get_user_model,authenticate

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
        email = self.cleaned_data.get('email')
        password = self.cleaned_data.get('password')
        user = None
        try:
            user_obj = User.objects.get(email=email)
            username = user_obj.username  # authenticate uses username
        except User.DoesNotExist:
            raise forms.ValidationError("Invalid email or password")

        user = authenticate(username=username, password=password)
        if user is None:
            raise forms.ValidationError("Invalid email or password")

        self.user = user
        return self.cleaned_data


