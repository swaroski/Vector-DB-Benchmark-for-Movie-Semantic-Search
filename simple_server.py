#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Set working directory to project root
project_root = Path(__file__).parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

print(f"Working directory: {os.getcwd()}")
print(f"Python path includes: {project_root}")

try:
    # Import the server app
    from ui.backend.server import app
    print("✅ Server imported successfully")
    
    # Start the server
    import uvicorn
    print("🚀 Starting server on http://localhost:8002")
    print("📁 Data directory:", project_root / "data")
    print("📁 Data exists:", (project_root / "data").exists())
    
    if (project_root / "data" / "movie.csv").exists():
        print("✅ movie.csv found")
    else:
        print("❌ movie.csv not found")
    
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
    
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    traceback.print_exc()