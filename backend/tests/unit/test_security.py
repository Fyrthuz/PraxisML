"""
Tests for app.core.security: JWT encoding/decoding and password hashing.
"""

from datetime import timedelta

import pytest
from jose import jwt

from app.core.security import (
    create_access_token,
    decode_token,
    get_password_hash,
    verify_password,
)


class TestPasswordHashing:
    def test_get_password_hash_returns_string(self):
        hashed = get_password_hash("my_password")
        assert isinstance(hashed, str)
        assert len(hashed) > 10

    def test_verify_password_correct(self):
        hashed = get_password_hash("my_password")
        assert verify_password("my_password", hashed) is True

    def test_verify_password_incorrect(self):
        hashed = get_password_hash("my_password")
        assert verify_password("wrong_password", hashed) is False

    def test_same_password_different_hash(self):
        h1 = get_password_hash("same_password")
        h2 = get_password_hash("same_password")
        assert h1 != h2
        assert verify_password("same_password", h1) is True
        assert verify_password("same_password", h2) is True


class TestCreateAccessToken:
    def test_create_access_token(self, monkeypatch):
        monkeypatch.setattr("app.core.security.settings.SECRET_KEY", "test_secret_key_12345")
        monkeypatch.setattr("app.core.security.settings.ALGORITHM", "HS256")
        monkeypatch.setattr("app.core.security.settings.ACCESS_TOKEN_EXPIRE_MINUTES", 60)

        token = create_access_token(subject="user_abc")
        assert isinstance(token, str)
        assert len(token.split(".")) == 3

    def test_create_access_token_with_custom_expiry(self, monkeypatch):
        monkeypatch.setattr("app.core.security.settings.SECRET_KEY", "test_secret_key_12345")
        monkeypatch.setattr("app.core.security.settings.ALGORITHM", "HS256")

        token = create_access_token(subject="user_abc", expires_delta=timedelta(minutes=30))
        assert isinstance(token, str)

    def test_create_access_token_with_extra_data(self, monkeypatch):
        monkeypatch.setattr("app.core.security.settings.SECRET_KEY", "test_secret_key_12345")
        monkeypatch.setattr("app.core.security.settings.ALGORITHM", "HS256")
        monkeypatch.setattr("app.core.security.settings.ACCESS_TOKEN_EXPIRE_MINUTES", 60)

        token = create_access_token(
            subject="user_abc",
            extra_data={"tenant_id": "tenant_xyz", "role": "admin"},
        )
        payload = jwt.decode(token, "test_secret_key_12345", algorithms=["HS256"])
        assert payload["sub"] == "user_abc"
        assert payload["tenant_id"] == "tenant_xyz"
        assert payload["role"] == "admin"


class TestDecodeToken:
    def test_decode_valid_token(self, monkeypatch):
        monkeypatch.setattr("app.core.security.settings.SECRET_KEY", "test_secret_key_12345")
        monkeypatch.setattr("app.core.security.settings.ALGORITHM", "HS256")
        monkeypatch.setattr("app.core.security.settings.ACCESS_TOKEN_EXPIRE_MINUTES", 60)
        monkeypatch.setattr("app.core.security.settings.EXTERNAL_AUTH_JWKS_URL", None)

        token = create_access_token(subject="user_abc")
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == "user_abc"
        assert "exp" in payload

    def test_decode_expired_token_returns_none(self, monkeypatch):
        monkeypatch.setattr("app.core.security.settings.SECRET_KEY", "test_secret_key_12345")
        monkeypatch.setattr("app.core.security.settings.ALGORITHM", "HS256")
        monkeypatch.setattr("app.core.security.settings.EXTERNAL_AUTH_JWKS_URL", None)

        token = create_access_token(subject="user_abc", expires_delta=timedelta(seconds=-10))
        payload = decode_token(token)
        assert payload is None

    def test_decode_wrong_key_returns_none(self, monkeypatch):
        monkeypatch.setattr("app.core.security.settings.ALGORITHM", "HS256")
        monkeypatch.setattr("app.core.security.settings.EXTERNAL_AUTH_JWKS_URL", None)

        token = jwt.encode({"sub": "test"}, "key_a", algorithm="HS256")
        monkeypatch.setattr("app.core.security.settings.SECRET_KEY", "key_b")
        payload = decode_token(token)
        assert payload is None

    def test_decode_corrupted_token_returns_none(self, monkeypatch):
        monkeypatch.setattr("app.core.security.settings.SECRET_KEY", "test_secret_key_12345")
        monkeypatch.setattr("app.core.security.settings.ALGORITHM", "HS256")
        monkeypatch.setattr("app.core.security.settings.ACCESS_TOKEN_EXPIRE_MINUTES", 60)
        monkeypatch.setattr("app.core.security.settings.EXTERNAL_AUTH_JWKS_URL", None)

        valid_token = create_access_token(subject="user_abc")
        corrupted = valid_token[:-10] + "INVALID"
        payload = decode_token(corrupted)
        assert payload is None
