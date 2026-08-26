"""
Sample Authentication Module
"""
import hashlib

class AuthService:
    def __init__(self):
        self.users = {"alice": "secret123", "bob": "qwerty456"}

    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate(self, username: str, password: str) -> bool:
        if username not in self.users:
            return False
        return self.users[username] == password

    def validate_token(self, token: str) -> bool:
        return bool(token and token.startswith("Bearer_"))
