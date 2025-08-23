#!/usr/bin/env python3

import sys
from pathlib import Path
import uvicorn

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the server
from ui.backend.server import app

if __name__ == "__main__":
    print(f"Starting server from {project_root}")
    print(f"Data directory: {project_root / 'data'}")
    print(f"Data directory exists: {(project_root / 'data').exists()}")
    
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)