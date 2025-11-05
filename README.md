# human-vision-ai
an web app which uses camera inputs based on that input our web page will give how the user is reacting and if he is tilting his head that will be shown 
Install these packages for your project:

fastapi
uvicorn (use uvicorn[standard])
numpy
opencv-python (or opencv-python-headless on headless servers)
aiofiles (required by FastAPI StaticFiles)
mediapipe (optional — only if you want face-landmark detection)

python -m venv .venv
.venv\Scripts\activate
pip install fastapi "uvicorn[standard]" numpy opencv-python aiofiles
# optional:
pip install mediapipe
# OR for headless:
pip install opencv-python-headless
