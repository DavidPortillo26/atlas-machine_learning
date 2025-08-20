<img width="348" height="145" alt="image" src="https://github.com/user-attachments/assets/ed4a3a05-758d-4d1f-ada8-16d74b01a4c0" />


QA Bot – Interactive Question Answering with Semantic Search


A quick look at the bot in action: asking questions and receiving answers from reference documents.

About the Developer

Hi, I’m David Portillo, an aspiring software engineer currently studying at Atlas School, with a focus on Machine Learning and Computer Science. My passion lies in building intelligent tools that make information easier to access.

LinkedIn:
https://www.linkedin.com/in/david-portillo26/

Portfolio Repository:
https://github.com/DavidPortillo26/davidportillo.github.io


Project Description

QA Bot is an interactive question-answering system designed to make searching through documentation as seamless as asking a question. Instead of scanning hundreds of lines of reference material, users can simply type a natural language question, and the bot will:

Use semantic search with sentence-transformers to find the most relevant reference document.

Apply a question-answering function to extract the most useful snippet.

Present the answer back in a conversational, human-friendly format.

This project bridges the gap between raw documentation and intuitive user interaction — turning static text files into a dynamic knowledge base.


Story of Development

This project began as a simple experiment: "Can I build a bot that answers questions using my notes?"

Step 1: I first built a basic Q&A function (0-qa.py) that could extract answers from a single document.

Step 2: I wanted more scalability, so I implemented semantic search to select the best document from an entire corpus.

Step 3: I tied everything together in an interactive shell (4-main.py) where users can ask multiple questions until they type exit, quit, or bye.

Through this journey, the bot evolved from a single-file experiment into a mini framework for document-driven Q&A.


Features Implemented

Load a corpus of Markdown (.md) files as reference documents
Perform semantic similarity search with sentence-transformers
Select the most relevant document per user query
Extract context-specific answers from that document
Interactive command-line interface with natural exit commands


Features Still To Be Implemented

Add a web-based frontend so users can ask questions via a simple UI
Support for multi-document answers (when one isn’t enough)
Enhanced error handling for missing or incomplete corpora
Integration with vector databases (like Pinecone or FAISS) for faster search


Challenges Faced

Dynamic Imports: Importing the QA module (0-qa.py) dynamically so I could build on earlier code without rewriting.

Semantic Search Tuning: Getting the SentenceTransformer to return the right document consistently required balancing embeddings and query handling.

Answer Accuracy: Extracting precise answers from free-form text was tricky — sometimes the bot would return too much context or miss subtle phrasing.

Documentation: Making sure others (even those who don’t read code) could understand what’s happening under the hood.

Each of these challenges shaped the project into what it is today.


Example Usage
$ python3 4-main.py
Q: When are PLD's?
A: PLD’s are scheduled for the third week of the month.
Q: Who can I reach out to for feedback?
A: Please contact your mentor via Slack or email.
Q: quit
A: Goodbye!

Inspiration

This project was inspired by my experience digging through endless class notes and documentation. I wanted something smarter than Ctrl+F, something closer to asking a friend who already knows where everything is.
