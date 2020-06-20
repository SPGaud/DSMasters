#!/usr/bin/env python
"""
Very simple HTTP server in python.
Usage::
    ./dummy-web-server.py [<port>]
Send a GET request::
    curl http://localhost
Send a HEAD request::
    curl -I http://localhost
Send a POST request::
    curl -d "foo=bar&bin=baz" http://localhost
"""
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer

import base64

def build_action_refill(where,what):
    text = "<action>\n"
    text += "<type>refill</type>\n"
    text += "<where>"+where+"</where>\n"
    text += "<what>"+base64.b64encode(what)+"</what>\n"
    text += "</action>\n"
    return text

def build_action_append(where,what):
    text = "<action>\n"
    text += "<type>append</type>\n"
    text += "<where>"+where+"</where>\n"
    text += "<what>"+base64.b64encode(what)+"</what>\n"
    text += "</action>\n"
    return text

def build_action_total(value):
    text = "<action>\n"
    text += "<type>total</type>\n"
    text += "<value>"+str(value)+"</value>\n"
    text += "</action>\n"
    return text

def build_action_reset():
    text = "<action>\n"
    text += "<type>reset</type>\n"
    text += "</action>\n"
    return text

    
class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):

        parts = self.path.split("?",1);        

        if (self.path == '/'):
            self.send_response(200)
            fname = 'till.html';
            file = open(fname,"r")
            text = file.read()
            self.send_header('Content-type', 'text/html')
        elif (self.path == '/till.css'):
            fname = 'till.css';
            file = open(fname,"r")
            text = file.read()
            self.send_header('Content-type', 'text/css')
            self.send_response(200)
        elif (self.path == '/till2.html'):
            self.send_response(200)
            fname = 'till2.html';
            file = open(fname,"r")
            text = file.read()
            self.send_header('Content-type', 'text/html')
        elif (self.path == '/till.js'):
            self.send_response(200)
            fname = 'till.js';
            file = open(fname,"r")
            text = file.read()
            self.send_header('Content-type', 'application/javascript')
        elif (parts[0] == '/action'):
            self.send_response(200)

            subtext = "";
            for p in parts[1].split("&"):
                subtext = subtext + p +"<br>";

            text  = '<?xml version="1.0" encoding="UTF-8"?>\n'
            text += "<response>\n"
            text += build_action_total(200);
            text += build_action_append("target", subtext);
            text += build_action_refill("title", "Change &pound;0.00");
            text += build_action_reset();
            text += "</response>\n"
            self.send_header('Content-type', 'application/xml')
        else:
            self.send_response(404)
            fname = '404.html';
            file = open(fname,"r")
            text = file.read()
            self.send_header('Content-type', 'text/html')
            
        self.end_headers()
        self.wfile.write(text)

    def do_HEAD(self):
        self._set_headers()
        
    def do_POST(self):
        self._set_headers()
        self.wfile.write("<html><body><h1>POST!</h1></body></html>")
        
def run(server_class=HTTPServer, handler_class=S, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print 'Starting httpd...'
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
