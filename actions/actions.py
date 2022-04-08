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
import os
from rasa_sdk.events import SlotSet, AllSlotsReset, EventType
from rasa_sdk.types import DomainDict

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
        game = tracker.get_slot("game_choice")
        if game != "Word of the day":
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
            simonFlag = random.randint(0,1)
            # read items
            # SimonDataItems = Path('data/SimonSays.txt').read_text().split('\n')

            #idx = random.randint(0, len(SimonDataItems)-1)
            test_data = ['fork', 'spoon', 'bottle', 'sports ball', 'rectangular object', 'cylindrical object', 'spherical object']
            idx = random.randint(0, len(test_data)-1)
            if test_data[idx][0] in ("a", "o", "e", "i", "u"):
                particle = "an "
            else:
                particle = "a "
            if simonFlag == 0:
                dispatcher.utter_message(text="Okay, Simon says, show me " + particle + str(test_data[idx]) + ".")
                return [SlotSet(key = "object_detection", value = "yes"), SlotSet(key = "item", value = str(test_data[idx]))]
            else:
                dispatcher.utter_message(text="Okay, Show me " + particle + str(SimonDataItems[idx]) + ".")
                return [SlotSet(key = "object_detection", value = "no"), SlotSet(key = "item", value = str(test_data[idx]))]
        elif game == "Word of the day":
            # if all words are learnt, move all learnt words back to database
            if os.stat("data/Word_of_the_day.txt").st_size == 0:
                    with open("data/Word_of_the_day.txt", "w") as f1, open("data/learnt_words.txt", "r+") as f2:
                        lines = f2.readlines()  
                        f1.writelines(lines)
                        f2.truncate(0)
                        f1.close()
                        f2.close()
            WordData = Path('data/Word_of_the_day.txt').read_text().split('\n')

            # randomly select a word
            randn = random.randint(0, len(WordData)-1)
            word = WordData[randn].split('/')[0]
            meaning = WordData[randn].split('/')[1]
            dispatcher.utter_message(text= "Okay let's learn a new word, repeat the word with me, are you ready?")
            return [SlotSet(key = "word", value = word), SlotSet(key = "word_meaning", value = meaning), SlotSet(key = "word_index", value = randn)]
        elif game == "Scissor paper stone":
            dispatcher.utter_message(text= "Okay let's play scissor paper stone, get ready, scissor, paper, stone.")
            return [SlotSet(key = "SPSflag", value = True)]
        elif game == "Pop the bubble":
            dispatcher.utter_message(text= "Okay let's play pop the bubble.")
            return [SlotSet(key = "PTBflag", value = True)]
        elif game == "Show me the number":
            dispatcher.utter_message(text= "Okay let's play show me the number.")
            return [SlotSet(key = "SMTNflag", value = True)]

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
            dispatcher.utter_message(text= "Okay, let me tell you a fun fact, " + str(FunFactData[randn]) + ", do you want to know another fun fact?")
            return []

class ActionChooseGame(Action):
    def name(self) -> Text:
        return "action_choose_game"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        randn = random.randint(0, 4)
        if randn == 0:
            dispatcher.utter_message(text="Let's play Simon says, shall we?")
            game = "Simon says"
        elif randn == 1:
            dispatcher.utter_message(text="Let's learn a word of the day, shall we?")
            game = "Word of the day"
        elif randn == 2:
            dispatcher.utter_message(text="Let's play scissor paper stone, shall we?")
            game = "Scissor paper stone"
        elif randn == 3:
            dispatcher.utter_message(text="Let's play pop the bubble, shall we?")
            game = "Pop the bubble"
        else:
            dispatcher.utter_message(text="Let's play show me the number, shall we?")
            game = "Show me the number"
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
            dispatcher.utter_message(text="Great, you are correct, do you want to play Simon says again?")
        else:
            if object_detection == "yes":
                dispatcher.utter_message(text="Oh no time out, do you want to play Simon says again?")
            else:
                dispatcher.utter_message(text="Oh no you are supposed to do nothing, do you want to play Simon says again?")
        return [SlotSet(key = "object_detection", value = "none"), SlotSet(key = "SimonsaysAns", value = "none")]

class AskForSlotAction(Action):
    def name(self) -> Text:
        return "action_ask_word_spoken"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        word = tracker.slots.get("word")
        repeat_counter = tracker.slots.get("repeat_counter")
        if repeat_counter == 1:
            dispatcher.utter_message(text="Repeat after me, " + word + ".")
        elif repeat_counter == 2:
            dispatcher.utter_message(text="Great, try to pronounce it again, " + word + ".")
        else:
            dispatcher.utter_message(text="Almost there, repeat one more time, " + word + ".")
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
                dispatcher.utter_message(text="Are you sure to quit the game, we haven't finished learning a word yet?")
                return []
        if repeat_counter < 3 and word != slot_value:
            return {"repeat_counter": repeat_counter+1, "word_spoken": None}
        else:
            dispatcher.utter_message(text="Well done, let me tell you the meaning of the word, "+ meaning + ", great, you learnt a new word today!")
            # move the learnt word to another database
            path = Path("data/learnt_words.txt")
            if not path.is_file():
                with open(path, "w+"):
                    pass
            idx = tracker.slots.get("word_index")
            with open("data/learnt_words.txt", "a") as f:
                if os.stat("data/learnt_words.txt").st_size == 0:
                    f.write(word + "/"+ meaning)
                else:
                    f.write("\n" + word + "/"+ meaning)
                f.close()
            with open("data/Word_of_the_day.txt", "r+") as f:
                lines = f.readlines()
                del lines[idx]  
                f.seek(0)
                f.truncate()
                f.writelines(lines)
                f.close()

            return {"repeat_counter": 0, "word_spoken": slot_value}

class ActionSPSmessage(Action):
    def name(self) -> Text:
        return "action_SPS_message"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        SPSmessage = tracker.slots.get("SPSmessage")
        dispatcher.utter_message(text=SPSmessage)
        return [SlotSet(key = "SPSflag", value = "none"), SlotSet(key = "SPSmessage", value = "none")]

class ActionPTBmessage(Action):
    def name(self) -> Text:
        return "action_PTB_message"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        status = tracker.slots.get("PTBstatus")
        duration =  int(tracker.slots.get("PTBduration"))
        if status == "1":
            message = "Awesome, you scored 10 points in " + str(duration) + " seconds, do you want to play again?"
        else:
            message = "Oh no time out please try to score 10 points in one minute, do you want to play again?"
        dispatcher.utter_message(text=message)
        return [SlotSet(key = "PTBstatus", value = "none"), SlotSet(key = "PTBduration", value = 0), SlotSet(key = "PTBflag", value = "none")]

class ActionSMTNmessage(Action):
    def name(self) -> Text:
        return "action_SMTN_message"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        status = tracker.slots.get("SMTNstatus")
        duration =  int(tracker.slots.get("SMTNduration"))
        if status == "1":
            message = "Awesome, you are able to show 5 numbers with hand gestures in " + str(duration) + " seconds, do you want to play again?"
        else:
            message = "Oh no time out, please try to show 5 numbers with hand gesture in one minute, do you want to play again?"
        dispatcher.utter_message(text=message)
        return [SlotSet(key = "SMTNstatus", value = "none"), SlotSet(key = "SMTNduration", value = 0), SlotSet(key = "SMTNflag", value = "none")]
