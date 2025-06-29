#!/usr/bin/env python3
"""
Run Dataset Curator with proper setup
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import uvicorn
    from curator import app
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Installing required packages...")
    os.system("pip install -r requirements.txt")
    print("✅ Installation completed. Please run the script again.")
    sys.exit(1)

def main():
    """Main function to run the curator."""
    print("🚀 Starting Dataset Curator...")
    print("📱 Server will start on http://localhost:8000")
    print("🎮 Use keyboard: 'A' to Accept, 'S' to Skip")
    print("💾 Accepted images will be saved to ../data/curated_dataset/")
    print("⚠️  Keep this terminal window open while using the curator!")
    print()
    print("Starting web server...")
    
    try:
        # Run the app
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except KeyboardInterrupt:
        print("\n🛑 Curator stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting curator: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main() 