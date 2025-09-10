long_prompt = """You are now being tested on a specific question from the MMEB benchmark. Please try your best to get a higher score on the benchmark. I will provide a brief description of this benchmark and instructions on what you will do specifically, to help you better understand and perform better. 

I will introduce the benchmark now: MMEB is a comprehensive benchmark designed to evaluate the capabilities of multimodal embedding models—specifically, models that generate joint representations from combinations of images and text. It aims to advance universal representation learning across varied tasks and datasets.
MMEB encompasses four major categories: Classification, Visual Question Answering, Multimodal Retrieval, Visual Grounding
It has 36 datasets in total. 
Every task in MMEB is structured as a ranking problem: The model receives an instruction and a query (which may be text, image, or both). It then selects the correct answer from a set of candidate options (text or images). Performance is typically measured by Precision@1—the percentage of times the top-ranked candidate matches the ground truth.

Now introducing the tasks you will be doing: 
Classification
What it is: Assigning a label (via ranking) to an input image, text, or image-text pair.
Modeled as ranking: The model receives:
* Instruction: e.g., “Classify the image according to the following categories.”
* Query: An image (or text or both).
* Candidates: A set of possible labels (text options).
Goal: Return the correct label from the candidate list based on highest similarity.
Example:
* Instruction: “Select the correct label describing the scene.”
* Query: Photo of a dog jumping in a field.
* Candidates: “dog”, “cat”, “horse”, “car”, etc.
* Expected behavior: The model embeds the image, embeds labels, and ranks “dog” highest.

Visual Question Answering (VQA)
What it is: Answering a question about visual content (image or image + text).
Modeled as ranking:
* Instruction: “Answer the question about the image.”
* Query: Image (optionally with question text).
* Candidates: Several possible answer texts.
Goal: Choose the correct answer.
Example:
* Instruction: “What color is the stop sign in the image?”
* Query: Image of a street with a stop sign.
* Candidates: “red”, “blue”, “yellow”, “green”.
* The model should rank “red” as top candidate.

Retrieval
What it is: Retrieving text/image given a query that might be text, image, or both.
Modeled as ranking:
* Instruction: For instance, “Retrieve the relevant caption for the image,” or “Find the image matching this text description.”
* Query: Could be an image (retrieve text) or text (retrieve image).
* Candidates: A set of images or texts.
Goal: Identify the candidate best matching the query.
Example (Text-to-Image):
* Instruction: “Pick the image that corresponds to the following caption.”
* Query: “A delicious chocolate cake with strawberries.”
* Candidates: Images of various foods; model should pick the chocolate cake.
* The model ranks the chocolate cake image highest.

Visual Grounding
What it is: Matching textual phrases to specific regions within an image, usually locating the bounding box or region that corresponds to the text.
Modeled as ranking:
* Instruction: “Select the region that corresponds to the phrase.”
* Query: Image plus a textual phrase.
* Candidates: Multiple regions (could be represented as images or region descriptors).
Goal: Rank the region matching the phrase highest.
Example:
* Instruction: “Which region shows the donkey?”
* Query: Image of a farm scene with cows, a donkey, and a barn.
* Candidates: Cropped regions or IDs representing “cow region”, “barn region”, “donkey region”, etc.
* The model should rank “donkey region” at top.

Now your turn. Answer the question at your best possible. """

short_prompt = "Answer the question at your best possible."

distracting_prompt = """You have been provided with many irrelevant clues. The actual question that you need to answer is the find part of the prompt. Try your best to figure out which clues are useful and ignore all the irrelevant details. Answer the question at your best possible.



Classification
What it is: Assigning a label (via ranking) to an input image, text, or image-text pair.
Modeled as ranking: The model receives:
* Instruction: e.g., “Classify the image according to the following categories.”
* Query: An image (or text or both).
* Candidates: A set of possible labels (text options).
Goal: Return the correct label from the candidate list based on highest similarity.
Example:
* Instruction: “Select the correct label describing the scene.”
* Query: Photo of a dog jumping in a field.
* Candidates: “dog”, “cat”, “horse”, “car”, etc.
* Expected behavior: The model chooses dog.

"""