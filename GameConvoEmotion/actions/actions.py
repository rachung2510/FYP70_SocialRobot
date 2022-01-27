# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, Action, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
import random
from pathlib import Path
from rasa_sdk.events import SlotSet, AllSlotsReset
from rasa_sdk.types import DomainDict
import time

class ActionTellJoke(Action):
    def name(self) -> Text:
        return "action_tell_joke"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

            JokeData = Path('data/Jokes.txt').read_text().split('\n')
            # randomly select a joke
            randn = random.randint(0, len(JokeData)-1)
            dispatcher.utter_message(text= str(JokeData[randn]))

class ActionEnterEmoMode(Action):
    def name(self) -> Text:
        return "action_enter_emo_mode"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        return [SlotSet(key = "emo_mode", value = True)]

class ActionExitEmoMode(Action):
    def name(self) -> Text:
        return "action_exit_emo_mode"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        return [SlotSet(key = "emo_mode", value = False)]

class ActionEnterGameMode(Action):
    def name(self) -> Text:
        return "action_enter_game_mode"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        return [SlotSet(key = "game_mode", value = True)]

class ActionExitGameMode(Action):
    def name(self) -> Text:
        return "action_exit_game_mode"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        return [SlotSet(key = "game_mode", value = None)]


class ActionPlayGame(Action):
    def name(self) -> Text:
        return "action_play_game"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        ## get the choice of the game
        game = tracker.get_slot("game_choice")

        if game == "Simon says":
            # whether to say 'Simon says' before hand
            simonFlag = random.randint(0,1)
            # read items
            SimonDataItems = Path('data/SimonSays.txt').read_text().split('\n')
            idx = random.randint(0, len(SimonDataItems)-1)
            if simonFlag == 0:
                dispatcher.utter_message(text="Simon says, show me " + str(SimonDataItems[idx]))
                return [SlotSet(key = "item", value = str(SimonDataItems[idx]))]
                #time.sleep(30)
            else:
                dispatcher.utter_message(text="Bring me "+ str(SimonDataItems[idx]))
                return [SlotSet(key = "item", value = "do nothing")]
                #time.sleep(90)
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
        else:
            dispatcher.utter_message(text="Let's learn a fun fact. Shall we?")
            game = "Fun fact"
        return [SlotSet(key = "game_choice", value = game)]
        
class ActionQuitGame(Action):
    def name(self) -> Text:
        return "action_quit_game"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [AllSlotsReset()]

class ActionCheckItem(Action):
    def name(self) -> Text:
        return "action_check_item"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        item = tracker.get_slot("item")
        item_none = tracker.get_slot("item_none")
        item_shown = tracker.get_slot("item_shown")
        # without "Simon says" before hand
        if item == "do nothing":
            if item_none == "do nothing":
                ans = True
            else:
                ans = False
        # with "Simon says" before hand
        else:
            if item == item_shown:
                ans = True
            else:
                ans = False
        # check answer
        if ans:
            dispatcher.utter_message(text="Great, you are correct!")
        else:
            dispatcher.utter_message(text="Oh no you are wrong!")
            return []
