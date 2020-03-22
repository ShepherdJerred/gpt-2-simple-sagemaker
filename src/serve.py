from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
import json

from generate import generate_response


def handle_ping(request):
    return Response({})


def handle_invocations(request):
    model_input = json.loads(request)
    model_output = generate_response(model_input)
    return Response({
        model_input: model_input,
        model_output: model_output
    })


with Configurator() as config:
    ping_route = 'ping'
    config.add_route(ping_route, '/ping')
    config.add_view(handle_ping, route_name=ping_route, renderer='json')

    invocations_route = 'invocations'
    config.add_route(invocations_route, '/invocations')
    config.add_view(handle_invocations, route_name=invocations_route,
                    renderer='json')

    app = config.make_wsgi_app()
server = make_server('0.0.0.0', 8080, app)
server.serve_forever()
