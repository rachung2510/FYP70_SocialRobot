# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import random
from pathlib import Path
from rasa_sdk.events import SlotSet
from rasa_sdk.events import FollowupAction

class ActionPlayGame(Action):
    def name(self) -> Text:
        return "action_play_game"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        ## get the choice of the game
        game = tracker.get_slot("game_choice")

        if game == "Simon says":
            SimonData = Path('data/SimonSays.txt').read_text().split('\n')
            # randomly select an item
            randn = random.randint(0, len(SimonData)-1)
            simonFlag = random.randint(0,1)
            if simonFlag == 0:
                dispatcher.utter_message(text="Bring me a "+ str(SimonData[randn]))
            else:
                dispatcher.utter_message(text="Simon says, bring me a "+ str(SimonData[randn]))
        elif game == "Word of the day":
            WordData = Path('data/Word_of_the_day.txt').read_text().split('\n')
            # randomly select a word
            randn = random.randint(0, len(WordData)-1)
            dispatcher.utter_message(text= str(WordData[randn]))
            # if pronounce incorrectly, repeat
            flag = 1
            counter = 0
            while (flag == 0 and counter != 2):
                dispatcher.utter_message(text= str(WordData[randn]) + ". Repeat with me.")
        elif game == "Fun fact":
            FunFactData = Path('data/FunFacts.txt').read_text().split('\n')
            # randomly select a fun fact
            randn = random.randint(0, len(FunFactData)-1)
            dispatcher.utter_message(text= str(FunFactData[randn]))
        return []

class ActionChooseGame(Action):
    def name(self) -> Text:
        return "action_choose_game"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        randn = random.randint(0, 2)
        if randn == 0:
            dispatcher.utter_message(text="Let's play Simon says. Shall we?")
            game = "Simon says"
        elif randn == 1:
            dispatcher.utter_message(text="Let's learn a word of the day. Shall we?")
            game = "Word of the day"
        else:
            dispatcher.utter_message(text="Let's learn a fun fact. Shall we?")
            game = "Fun fact"
        return [SlotSet(key = "game_choice", value = game)]
        