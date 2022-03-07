# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
import random
from pathlib import Path
from rasa_sdk.events import SlotSet, AllSlotsReset
from rasa_sdk.types import DomainDict

class ActionDisallowNonsense(Action):
    def name(self) -> Text:
        return "action_disallow_nonsense"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [SlotSet(key = "disallow_nonsense", value = True)]

class ActionAllowNonsense(Action):
    def name(self) -> Text:
        return "action_allow_nonsense"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [SlotSet(key = "disallow_nonsense", value = False)]
        
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
            return []

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
        dispatcher.utter_message(text="Okay, let me know if you want to play any game.")
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
            simonFlag = 1 #random.randint(0,1)
            # read items
            # SimonDataItems = Path('data/SimonSays.txt').read_text().split('\n')
            # idx = random.randint(0, len(SimonDataItems)-1)
            test_data = ['fork']#, 'spoon', 'bottle']
            idx = random.randint(0, len(test_data)-1)
            if simonFlag == 0:
                dispatcher.utter_message(text="Okay. Simon says, show me " + str(test_data[idx]))
                return [SlotSet(key = "object_detection", value = "yes"), SlotSet(key = "item", value = str(test_data[idx]))]
            else:
                dispatcher.utter_message(text="Okay. Show me " + str(test_data[idx]))
                return [SlotSet(key = "object_detection", value = "no"), SlotSet(key = "item", value = str(test_data[idx]))]
        elif game == "Word of the day":
            WordData = Path('data/Word_of_the_day.txt').read_text().split('\n')
            
            # randomly select a word
            randn = random.randint(0, len(WordData)-1)
            word = WordData[randn].split('/')[0]
            meaning = WordData[randn].split('/')[1]
            dispatcher.utter_message(text= "Okay let's learn a new word. Repeat the word with me. Are you ready?")
            return [SlotSet(key = "word", value = word), SlotSet(key = "word_meaning", value = meaning)]
        elif game == "Scissor paper stone":
            dispatcher.utter_message(text= "Okay let's play scissor paper stone. Tell me your choice")
            return []

class ActionTellFunFact(Action):
    def name(self) -> Text:
        return "action_tell_fun_fact"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            # FunFactData = Path('data/FunFacts.txt').read_text().split('\n')
            with open('data/FunFacts.txt', encoding='utf-8') as f:
                FunFactData = f.readlines()
            # randomly select a fun fact
            randn = random.randint(0, len(FunFactData)-1)
            dispatcher.utter_message(text= "Okay, let me tell you a fun fact. " + str(FunFactData[randn]) + " Do you want to know another fun fact?")
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
        ans = tracker.get_slot("SimonsaysAns")
        object_detection = tracker.get_slot("object_detection")
        # check answer
        if ans == True:
            dispatcher.utter_message(text="Great, you are correct! Do you want to play Simon says again?")
        else:
            if object_detection == "yes":
                dispatcher.utter_message(text="Oh no time out! Do you want to play Simon says again?")
            else:
                dispatcher.utter_message(text="Oh no you are supposed to do nothing! Do you want to play Simon says again?")
        return [SlotSet(key = "object_detection", value = "none"), SlotSet(key = "SimonsaysAns", value = "none")]

class ValidateItemForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_item_form"

    def validate_item_donothing(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        item = tracker.slots.get("item")
        if slot_value != 'none':
            if slot_value == "do nothing":
                dispatcher.utter_message(text="Great, you are correct! Do you want to play Simon says again?")
            else:
                dispatcher.utter_message(text="Oh no, you are supposed to say do nothing since I never say Simon says! Do you want to play Simon says again?")
            return {"item_donothing": slot_value}
        return {"item_donothing": None}

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
                return []
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
        if choice == "scissor":
            if computerChoice == "Scissor":
                dispatcher.utter_message(text="It is a tie. My choice was scissor too." + playagain)
            elif computerChoice == "Paper":
                dispatcher.utter_message(text="Yay you win. My choice was paper." + playagain)
            else:
                dispatcher.utter_message(text="Oh no you lose. My choice was stone." + playagain)
        elif choice == "paper":
            if computerChoice == "Scissor":
                dispatcher.utter_message(text="Oh no you lose. My choice was scissor" + playagain)
            elif computerChoice == "Paper":
                dispatcher.utter_message(text="It is a tie. My choice was paper too." + playagain)
            else:
                dispatcher.utter_message(text="Yay you win. My choice was stone." + playagain)
        elif choice == "stone":
            if computerChoice == "Scissor":
                dispatcher.utter_message(text="Yay you win. My choice was scissor." + playagain)
            elif computerChoice == "Paper":
                dispatcher.utter_message(text="Oh no you lose. My choice was paper." + playagain)
            else:
                dispatcher.utter_message(text="It is a tie. My choice was stone too." + playagain)
        return []