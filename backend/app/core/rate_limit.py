"""
Rate Limiter compartido (slowapi).

Se importa en los routers que necesiten rate limiting
y se registra en la app en main.py.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
