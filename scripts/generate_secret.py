#!/usr/bin/env python
"""Generate a secure SECRET_KEY for .env"""
import secrets
print(secrets.token_hex(32))
