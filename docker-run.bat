@echo off
docker run -it -p 7860:7860 voice-studio python -c "import app; app.interface.launch(server_name='0.0.0.0', server_port=7860)"