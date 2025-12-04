#!/usr/bin/env python3
"""
Simple HTTP Server for AI Hiring Portal UI
Serves the static HTML/CSS/JS files from the ui directory
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Configuration
PORT = 8000
UI_DIRECTORY = "src/ui"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve files from the ui directory"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Add headers to prevent caching during development
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def log_message(self, format, *args):
        # Custom logging format
        print(f"📡 {self.address_string()} - {format % args}")

def main():
    """Start the HTTP server"""
    
    # Check if UI directory exists
    if not os.path.exists(UI_DIRECTORY):
        print(f"❌ Error: UI directory '{UI_DIRECTORY}' not found!")
        print(f"   Current directory: {os.getcwd()}")
        return
    
    print("🚀 AI HIRING PORTAL - LOCAL SERVER")
    print("=" * 60)
    print(f"📂 Serving files from: {os.path.abspath(UI_DIRECTORY)}")
    print(f"🌐 Server running at: http://localhost:{PORT}")
    print(f"📄 Login page: http://localhost:{PORT}/login.html")
    print("=" * 60)
    print("\n👤 Demo Credentials:")
    print("   Recruiter: username='recruiter', password='recruiter123'")
    print("   Candidate: username='candidate', password='candidate123'")
    print("   Admin: username='admin', password='admin123'")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    # Create server
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        try:
            # Open browser automatically
            webbrowser.open(f"http://localhost:{PORT}/login.html")
            print("✅ Server started successfully!")
            print(f"🌐 Opening browser at http://localhost:{PORT}/login.html\n")
            
            # Serve forever
            httpd.serve_forever()
            
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped by user")
            print("👋 Goodbye!")

if __name__ == "__main__":
    main()
