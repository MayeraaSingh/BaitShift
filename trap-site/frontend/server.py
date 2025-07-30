#!/usr/bin/env python3
import http.server
import socketserver
import os
from urllib.parse import urlparse

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # If the path is /securechat, serve index.html
        if parsed_path.path == '/securechat':
            self.path = '/index.html'
        
        # Call the parent method to handle the request
        return super().do_GET()

if __name__ == "__main__":
    PORT = 8000
    
    # Change to the directory containing the HTML files
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}")
        print(f"Securechat at http://localhost:{PORT}/securechat")
        httpd.serve_forever()
