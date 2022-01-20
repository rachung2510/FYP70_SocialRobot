# Merged Casual Convo & Emotion Recognition branch
Main file ***tts_link_test.py*** gets emotion models from **emotion_data/** and calls ***emotion_recognition.py*** for the emotion prediction.
1. Open a terminal and start virtual environment with ```myvenv\Scripts\activate```.
2. Enter **CasualConvoTTS/** directory with ```cd FYP70_SocialRobot\CasualConvoTTS```.
2. Start the RASA server with ```rasa run -m models --endpoints endpoints.yml --port 5002 --credentials credentials.yml```.
3. Open another terminal and start virtual environment again.
4. Enter **tts_model/** directiory with ```cd FYP70_SocialRobot\CasualConvoTTS\tts_model```.
5. Run ```python tts_link_test.py```.
