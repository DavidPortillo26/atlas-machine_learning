#!/usr/bin/env python3
"""
Interactive question-answering loop using a reference text.
"""

import importlib.util
import sys
from typing import Optional

# Dynamically import 0-qa.py
spec = importlib.util.spec_from_file_location("qa_module", "./0-qa.py")
qa_module = importlib.util.module_from_spec(spec)
sys.modules["qa_module"] = qa_module
spec.loader.exec_module(qa_module)

# Get the question_answer function
question_answer = qa_module.question_answer

EXIT_COMMANDS = {"exit", "quit", "goodbye", "bye"}

def answer_loop(reference: str) -> None:
    """
    Continuously asks the user for questions and provides answers.

    Parameters:
        reference (str): The text to search for answers.
    """
    while True:
        user_input = input("Q: ").strip()

        if user_input.lower() in EXIT_COMMANDS:
            print("A: Goodbye")
            break

        answer: Optional[str] = question_answer(user_input, reference)
        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
