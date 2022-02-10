# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, Action, FormValidationAction
from Object_Detection import SimonSays_item
from rasa_sdk.executor import CollectingDispatcher
import random
from pathlib import Path
from rasa_sdk.events import SlotSet, AllSlotsReset, FollowupAction, ActiveLoop
from rasa_sdk.types import DomainDict

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

        return [AllSlotsReset()]

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
            # simonFlag = random.randint(0,1)
            simonFlag = 0
            # read items
            # SimonDataItems = Path('data/SimonSays.txt').read_text().split('\n')
            # idx = random.randint(0, len(SimonDataItems)-1)
            test_data = ['fork', 'spoon', 'sports ball', 'bottle']
            idx = random.randint(0, len(test_data)-1)
            if simonFlag == 0:
                dispatcher.utter_message(text="Okay. Simon says, show me " + str(test_data[idx]))
                return [SlotSet(key = "item", value = str(test_data[idx]))]
            else:
                dispatcher.utter_message(text="Okay. Bring me "+ str(test_data[idx]))
                return [SlotSet(key = "item", value = "do nothing")]
        elif game == "Word of the day":
            WordData = Path('data/Word_of_the_day.txt').read_text().split('\n')
            
            # randomly select a word
            randn = random.randint(0, len(WordData)-1)
            word = WordData[randn].split('/')[0]
            meaning = WordData[randn].split('/')[1]
            dispatcher.utter_message(text= "Okay let's learn a new word. Repeat the word with me. Are you ready?")
            return [SlotSet(key = "word", value = word), SlotSet(key = "word_meaning", value = meaning)]
        elif game == "Scissor paper stone":
            dispatcher.utter_message(text= "Okay let's play scissor paper stone. Tell me you choice")
            return []

class ActionTellFunFact(Action):
    def name(self) -> Text:
        return "action_tell_fun_fact"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            FunFactData = Path('data/FunFacts.txt').read_text().split('\n')
            # randomly select a fun fact
            randn = random.randint(0, len(FunFactData)-1)
            dispatcher.utter_message(text= "Okay, let me tell you a fun fact." + str(FunFactData[randn]) + " Do you want to know another fun fact?")
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
        
class ActionCheckItem(Action):
    def name(self) -> Text:
        return "action_check_item"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        item = tracker.get_slot("item")
        item_none = tracker.get_slot("item_none")
        # without "Simon says" before hand
        if item == "do nothing":
            if item_none == "do nothing":
                ans = True
            else:
                ans = False
        # with "Simon says" before hand
        else:
            ans = SimonSays_item(item)
        # check answer
        if ans:
            dispatcher.utter_message(text="Great, you are correct! Do you want to play Simon says again?")
        else:
            dispatcher.utter_message(text="Oh no time out! Do you want to play Simon says again?")
        return []

class ValidatePronunciationForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_pronunciation_form"

    def validate_word_spoken(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        word = tracker.slots.get("word")
        meaning = tracker.slots.get("word_meaning")
        repeat_counter = tracker.slots.get("repeat_counter")
        if slot_value == "quit":
                dispatcher.utter_message(text="Are you sure to quit the game? We haven't finished learning a word yet.")
                return []#ActiveLoop(None)]#, AllSlotsReset()]
                # return [SlotSet("stop_word_form", True)]
        if repeat_counter < 3 and word != slot_value:
                if repeat_counter == 0:
                    dispatcher.utter_message(text="Repeat the word with me.")
                elif repeat_counter == 1:
                    dispatcher.utter_message(text="Great, try to pronounce it again!")
                else:
                    dispatcher.utter_message(text="Almost there. Repeat one more time with me.")
                return {"repeat_counter": repeat_counter+1, "word_spoken": None}
        else:
            dispatcher.utter_message(text="Well done! Let me tell you the meaning of the word. "+ meaning + "Do you want to learn another word?")
            return {"repeat_counter": 0, "word_spoken": slot_value}

class ActionCheckSPS(Action):
    def name(self) -> Text:
        return "action_check_SPS"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        choice = tracker.get_slot("SPSchoice")
        choicelist = ["Scissor", "Paper", "Stone"]
        randn = random.randint(0, 2)
        computerChoice = choicelist[randn]
        playagain = "Do you want to play again?"
        if choice == "Scissor":
            if computerChoice == "Scissor":
                dispatcher.utter_message(text="It is a tie. My choice was scissor too." + playagain)
            elif computerChoice == "Paper":
                dispatcher.utter_message(text="Yay you win. My choice was paper." + playagain)
            else:
                dispatcher.utter_message(text="Oh no you lose. My choice was stone." + playagain)
        elif choice == "Paper":
            if computerChoice == "Scissor":
                dispatcher.utter_message(text="Oh no you lose. My choice was scissor" + playagain)
            elif computerChoice == "Paper":
                dispatcher.utter_message(text="It is a tie. My choice was paper too." + playagain)
            else:
                dispatcher.utter_message(text="Yay you win. My choice was stone." + playagain)
        elif choice == "Stone":
            if computerChoice == "Scissor":
                dispatcher.utter_message(text="Yay you win. My choice was scissor." + playagain)
            elif computerChoice == "Paper":
                dispatcher.utter_message(text="Oh no you lose. My choice was paper." + playagain)
            else:
                dispatcher.utter_message(text="It is a tie. My choice was stone too." + playagain)
        return []