import re
import json
from urllib.parse import urlparse

class RequestInfo:
    def __init__(self, method, url, body, headers=None, **kwargs):
        self.method = method
        self.url = url
        self.body = body
        self.headers = headers
        assert len(kwargs) < 15, "Too many arguments"
        self.__dict__.update(kwargs)

    def __str__(self):
        # return json.dumps(self.request)
        return self.url
    
    def dump_json(self):
        return json.dumps(self.__dict__)

requestionInfo = RequestInfo("POST", "/tienda1/miembros/editar.jsp", "modo=registro&login=stainbac&password=RAstRiLLa&nombre=Iber%E1&apellidos=Sejas+Escriba&email=taggart.jobert3%40grupodescansoyhogar.us&dni=06184698K&direccion=Carretera+De+Murcia+59+8%3FD&ciudad=Huete&cp=23340&provincia=Salamanca&ntc=6789852392757283&B1=Registrar")

def is_form_urlencoded(body):
    """
    Checks if the given body string is in form-urlencoded format.
    """
    pattern = r'^[\w.%+]+=[\S]*'
    return bool(re.match(pattern, body))

def get_http_level_split(req: RequestInfo):
    """
    Splits an HTTP request into minimal semantic units.
    """
    # Parse the URL to extract its components
    parsed = urlparse(req.url)
    
    # Split the URL path into parts, adding a '/' prefix to each segment
    path_parts = parsed.path.split('/')
    url_list = ['/' + part for part in path_parts if part]
    
    # Split the query string into a list of key-value pairs (if any)
    query_list = parsed.query.split('&')
    
    # Check if the request body is form-urlencoded and split it accordingly
    if is_form_urlencoded(req.body):
        body_list = req.body.split('&')
    else:
        # If the body is not form-encoded, treat the entire body as one element
        body_list = [req.body] if req.body else []

    if len(query_list) == 1 and query_list[0] == '':
        query_list = []

    group = [req.method] + url_list + (query_list if query_list else []) + body_list
    group = [item for item in group if item.strip()]
    
    return group

def get_http_level_split_furl(req: RequestInfo):
    """
    Splits an HTTP request into minimal semantic units.
    This function is tailored for the PKDD dataset, where the entire path 
    is treated as a single component for attack annotations.

    """
    # Parse the URL to extract components
    parsed = urlparse(req.url)

    # Use the entire URL path as a single element
    url_part = parsed.path

    # Split the query string into key-value pairs (if any)
    query_list = parsed.query.split('&')

    # Check if the request body is form-urlencoded and split it accordingly
    if is_form_urlencoded(req.body):
        body_list = req.body.split('&')
    else:
        # If the body is not form-encoded, treat the entire body as one element
        body_list = [req.body] if req.body else []

    if len(query_list) == 1 and query_list[0] == '':
        query_list = []

    group = [req.method] + [url_part] + (query_list if query_list else []) + body_list
    group = [item for item in group if item.strip()]
    
    return group

print (get_http_level_split(requestionInfo))

