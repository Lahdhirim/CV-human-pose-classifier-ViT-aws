"""
Specific file required for streamlit prototype deployment
"""

import subprocess
import sys
import time

if __name__ == "__main__":

    print("Starting FastAPI server...")
    fastapi_process = subprocess.Popen([sys.executable, "src/web_app/server.py"])

    time.sleep(5)

    print("Starting Streamlit interface...")
    subprocess.run(["streamlit", "run", "src/web_app/interface.py"])
