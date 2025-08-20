#!/usr/bin/env python3
"""
Interactive question-answering loop using a reference text.

This script allows a user to type questions and receive answers from a
reference text. It uses the question_answer function from 0-qa.py to find
answers. The loop continues until the user types a command to exit.
"""

import importlib.util
import sys
from typing import Optional

# Dynamically import the 0-qa.py module, which contains the question_answer
# function. This is done to allow importing a module whose name starts
# with a digit.
spec = importlib.util.spec_from_file_location("qa_module", "./0-qa.py")
qa_module = importlib.util.module_from_spec(spec)
sys.modules["qa_module"] = qa_module
spec.loader.exec_module(qa_module)

# Get the question_answer function from the imported module
question_answer = qa_module.question_answer

# Commands that will terminate the loop. Comparison is case-insensitive.
EXIT_COMMANDS = {"exit", "quit", "goodbye", "bye"}

def answer_loop(reference: str) -> None:
    """
    Continuously asks the user for questions and provides answers.

    Parameters:
        reference (str): The text to search for answers.

    How it works:
        1. Prompts the user with "Q:" to type a question.
        2. Checks if the input is an exit command. If so, prints
           "A: Goodbye" and stops the loop.
        3. Uses question_answer to try to find an answer in the reference.
        4. If an answer is found, prints "A: <answer>".
        5. If no answer is found, prints a fallback message:
           "A: Sorry, I do not understand your question."
        6. Repeats until an exit command is entered.
    """
    while True:
        # Prompt user for input and remove leading/trailing whitespace
        user_input = input("Q: ").strip()

        # Check if user wants to exit the loop
        if user_input.lower() in EXIT_COMMANDS:
            print("A: Goodbye")
            break

        # Find answer from the reference text using question_answer
        answer: Optional[str] = question_answer(user_input, reference)

        # Print the answer if found, otherwise print a fallback message
        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
