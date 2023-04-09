# EDU-Bot - An Eductional Social Robot for Toddlers

## Overview 
This is the undergraduate final year project for Group 70 under the Department of Electrical Engineering at the National University of Singapore.

In this project, an educational social robot, EDU-Bot, was developed with the aim to teach and guild young children between ages 4 to 6 to communicate effectively through social interaction and fun yet informative games. 

## Functions
The EDU-Bot uses natural language processing (NLP), speech processing, object detection and emotion detection its two main functions:
1. Casual conversation - the robot converses naturally with the child and reacts appropriately to the child's facial expressions.
2. Interactive games - the robot engages the child in a variety of fun games that utilize object detection to increase interactivity.

Its games include: Pop the bubble, Simon Says, Scissors Paper Stone (the Singaporean name for Rock Paper Scissors), Show Me the Number and Word of the Day

## Modules
### Natural Language Processing
The open-source [Rasa](https://github.com/RasaHQ/rasa) library was selected to handle natural language processing for casual conversation. Rasa classifies inputs according to user intents to generate appropriate responses. The nuances of Singaporean english, which is slightly different from standard English, was also taken into account. An example conversation is shown below, with ```USR``` and ```SYS``` denoting the child and robot accordingly:
```
USR >>> what is your name
SYS >>> I have a name, and that's edubot!
USR >>> I am happy
SYS >>> Oh I can see how great you feel from here!
USR >>> I got sticker
SYS >>> Glad to hear it!
USR >>> My friend drew for me
SYS >>> I want to be your friend too!
```

### Object Detection
Object detection was integrated to make games more interactive and educational for the child. Hands detection in games like Pop the Bubble, Show Me the Number and Scissors Paper Stone used Google's Mediapipe library for rapid and accurate hand detection at close proximity. In the game Simon Says, children were tasked to bring objects according to what "Simon says"; the YOLOv4 model was used for detecting household objects like balls, bottles, spoons, or object shapes like "rectangular". 

### Emotion Recognition
During casual conversation, the robot constantly detects the child's facial expressions for signs of happiness, sadness, etc. When a "non-neutral" expression is detected, the robot will output an appropriate response, e.g. "You look happy! Did something good happen?"

