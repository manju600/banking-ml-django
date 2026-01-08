"""
ASGI config for Machine_Learning_to_Detect_Periodicity_in_Transactional_Banking_Data project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Machine_Learning_to_Detect_Periodicity_in_Transactional_Banking_Data.settings')

application = get_asgi_application()
