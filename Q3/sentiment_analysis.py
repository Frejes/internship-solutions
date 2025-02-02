from transformers import pipeline

# Listing of the possible sentiments
sentiment_labels = ["Love", "Laughter", "Anger", "Compassion", "Disgust", "Fear", "Heroism", "Wonder", "Peace"]

# Loading the zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_top_sentiments(text):
    result = classifier(text, candidate_labels=sentiment_labels)

    # Sorting results by confidence scores
    sorted_result = sorted(zip(result['labels'], result['scores']), key=lambda x: x[1], reverse=True)

    # Getting the top 3 sentiments with percentages
    top_sentiments = {sorted_result[i][0]: f"{int(sorted_result[i][1] * 100)}%" for i in range(3)}

    return top_sentiments

# Taking input from the user
text = input("Enter your text for sentiment analysis: ")

# Getting sentiment analysis result
output = get_top_sentiments(text)

# Displaying result
print("\nTop 3 Sentiments:")
for sentiment, confidence in output.items():
    print(f"{sentiment}: {confidence}")
