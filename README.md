# Merged Casual Convo & Emotion Recognition branch
Combines the casual convo and emotion prediction parts to allow the user to input messages to the bot, but the bot will also be looking at the user's emotions and will output an appropriate response when a non-neutral emotion is detected, even if the user has not inputted anything.

Main file ***tts_link_test.py*** gets emotion models from **emotion_data/** and calls ***emotion_recognition.py*** for the emotion prediction.
1. Open a terminal and start virtual environment with ```myvenv\Scripts\activate```.
2. Enter **CasualConvoTTS/** directory with ```cd FYP70_SocialRobot\CasualConvoTTS```.
2. Start the RASA server with ```rasa run -m models --endpoints endpoints.yml --port 5002 --credentials credentials.yml```.
3. Open another terminal and start virtual environment again.
4. Enter **tts_model/** directiory with ```cd FYP70_SocialRobot\CasualConvoTTS\tts_model```.
5. Run ```python tts_link_test.py```.
