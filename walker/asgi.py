"""
ASGI config for walker project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""
import os
from django.core.asgi import get_asgi_application
from django.core.handlers.asgi import ASGIHandler

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'walker.settings')


class LifespanApp:

  def __init__(self, app):
    self.app = app

  async def __call__(self, scope, receive, send):
    if scope['type'] == 'lifespan':
      while True:
        message = await receive()
        if message['type'] == 'lifespan.startup':
          await send({'type': 'lifespan.startup.complete'})
        elif message['type'] == 'lifespan.shutdown':
          await send({'type': 'lifespan.shutdown.complete'})
          return
    else:
      await self.app(scope, receive, send)


application = LifespanApp(get_asgi_application())
