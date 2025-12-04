#!/usr/bin/env python3
"""
Startup script for the Smart Agriculture FastAPI backend
Run this script to start the unified backend server
"""

import uvicorn
from unified_backend import app

if __name__ == "__main__":
    print("🌱 Starting Smart Agriculture Backend Server...")
    print("📍 Server will be available at: http://127.0.0.1:8000")
    print("📖 API documentation at: http://127.0.0.1:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        "unified_backend:app", 
        host="127.0.0.1", 
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
