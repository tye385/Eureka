For the purpose of this project I created over 70 scripts and 10 colab files.  Not all code was used in the final model, so i have included the most relevant scripts and will describe them below:

Generate_titles.py shows how I represented the primary dataset data structure that was used throughout the title labeling process.  In the dataset, each .json object has 3 fields:
-ID: unique identifier mapping the article in both url and file structureTitle: Title of the wikipedia Article
Text: The content of the article
I opted for a dict object for fast access 
In this example, the IDs of each entry in the dataset are used as keys, and the titles are used as values.  I would use this schema for both Chat-GPT labeling, and labeling used the model trained through transfer learning.

label_titles_chatgpt.py illustrates how we aquired labeled data from chat-gpt through the openai api.  
We used threading to maximize the rate of labeling, while using tiktoken to calculate the cost of individual experiments.
The results of this program were used as the basis for training a much simpler classifier to distinguish between scientific and non-scientific articles

In the grid_search_transfer_learning.py file, we trained a BERT classifier to distinuish between scientific and non-scientific articles.  
We used the focal loss function to place emphasis on the minority class.  This is because it is far more important to inal 6.5 million articles, it is easier to prune false positives from a small dataset than it is to find missinroup.
The optimal values achieved were an alpha value of 30 and a gamma value of 2.
The resulting classifier had an accuracy of 96.5%, however more importantly it had a recall of 82.22% for the minority class IE scientific articles.

We then pruned the false positives with a program quite similar to label_titles_chatgpt.py, but on the entire remaining dataset.  

filter_dataset.py was then used to move the scientific articles from the full corpus to a folder containing only scientific articles
These articles were then used in train_gpt.ipynb to imbue a gpt2 model with the knowledge of the wikipedia corpus.  It was our hope that this would give GPT2 a basis for understanding the topics in the trivia test sets.

While our next objective was to generate test questions using Chat-GPT, we found that an existing set was available on the compeition forums.  The questions were generated in an identical way to our planned methods, so we utilized this resource instead of reinventing the wheel and incurring the cost of API calls to Chat-GPT.
In generate_context.py, we took the example questions and generated the prompts to be used in the fine tuning process.
We used a vectorized database of our dataset, and selected the most relevant title, then the most relevant passage from the article.  Each context is of variable length, due to the variable length of the questions.  

Finally, we fine tuned the model on these training questions in fine_tune_gpt.ipynb
We froze the models weights, and used PEFT to load an adapter on top of our model for fine tuning.  This makes sure that we do not lose the context learned in the first training set, while also allowing the model to focus on the task at hand: answering questions.

Lastly in server.py, we deployed the model through a flask application.  This model is accessible via http requests, and responds to questions with the correct answer.  We use the same strategy of leveraging our vectorized database to provide context for the model, and use the same maximum length as during training.

