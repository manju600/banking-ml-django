"""
WSGI config for Machine_Learning_to_Detect_Periodicity_in_Transactional_Banking_Data project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Machine_Learning_to_Detect_Periodicity_in_Transactional_Banking_Data.settings')

application = get_wsgi_application()
