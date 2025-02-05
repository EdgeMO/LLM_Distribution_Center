import re

# The input string (you would replace this with the actual input)
input_string = "['Please answer with the letter of the correct answer.\n\nIan volunteered to eat Dennis\'s menudo after already having a bowl because _ despised eating intestine.\nA) Ian\nB) Dennis\nPrint only a single choice  from ""A"" or ""B"" without explanation. Answer:\nB\n\nPlease answer with the letter of the correct answer.\n\nIan volunteered to eat Dennis\'s menudo after already having a bowl because _ enjoyed eating intestine.\nA) Ian\nB) Dennis\nPrint only a single choice  from ""A"" or ""B"" without explanation. Answer:\nA\n\nPlease answer with the letter of the correct answer.\n\nHe never comes to my home, but I always go to his house because the _ is smaller.\nA) home\nB) house\nPrint only a single choice  from ""A"" or ""B"" without explanation. Answer:\nA\n\nPlease answer with the letter of the correct answer.\n\nHe never comes to my home, but I always go to his house because the _ is bigger.\nA) home\nB) house\nPrint only a single choice  from ""A"" or ""B"" without explanation. Answer:\nB\n\nPlease answer with the letter of the correct answer.\n\nKyle doesn\'t wear leg warmers to bed, while Logan almost always does. _ is more likely to live in a colder climate.\nA) Kyle\nB) Logan\nPrint only a single choice  from ""A"" or ""B"" without explanation. Answer:\nB\n\nJennifer dragged Felicia along to a self help workshop about how to succeed, because _ wanted some company.\nA) Jennifer\nB) Felicia\nPrint only a single choice  from ""A"" or ""B"" without explanation. Answer:']"


# Split the content by 'Answer:'
parts = input_string.split('Answer:')

# The first part is the first question
questions = [parts[0].strip()]
answers = []

# Process the remaining parts
for part in parts[1:]:
    # Split each part into answer and next question
    split_part = part.split('\n', 1)
    
    if len(split_part) > 1:
        answer, next_question = split_part
        answers.append(answer.strip())
        questions.append(next_question.strip())
    else:
        # This handles the case where there's an answer without a following question
        answers.append(split_part[0].strip())

# Print the question-answer pairs
for i, (question, answer) in enumerate(zip(questions, answers), 1):
    print(f"Question {i}:")
    print(question)
    print(f"Answer: {answer}\n")

# If there's an extra question without an answer, print it
if len(questions) > len(answers):
    print(f"Question {len(questions)}:")
    print(questions[-1])
    print("Answer: [No answer provided]\n")