import nltk
nltk.download('wordnet')
import torch
import nltk
from transformers import pipeline, RobertaTokenizer, RobertaModel
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('wordnet')

# Load RoBERTa tokenizer and model
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")
metaphor_detector = pipeline("ner", model="CreativeLang/metaphor_detection_roberta_seq", tokenizer="CreativeLang/metaphor_detection_roberta_seq")

# Function to get senses of a word from WordNet
def get_word_senses(word):
    return wn.synsets(word)

# Function to calculate cosine similarity between embeddings
def calculate_similarity(context_embedding, sense_embedding):
    context_embedding = context_embedding.detach().numpy()
    sense_embedding = sense_embedding.detach().numpy()

    # Reshape embeddings to 2D arrays
    context_embedding = context_embedding.reshape(context_embedding.shape[1], -1)
    sense_embedding = sense_embedding.reshape(sense_embedding.shape[1], -1)

    return cosine_similarity(context_embedding, sense_embedding)

# Function to perform Word Sense Disambiguation (WSD) on a single input sentence
def perform_wsd(input_sentence):
    # Step 1: Extract metaphoric words using NER pipeline
    tagged_sentence = metaphor_detector(input_sentence)

    # Extract metaphoric words
    metaphoric_words = [word_info["word"] for word_info in tagged_sentence if word_info["entity"] == "LABEL_1"]

    # Step 2: Find context embeddings for metaphoric words and perform WSD
    for word in metaphoric_words:
        # Tokenize the word using RoBERTa tokenizer
        encoded_word = roberta_tokenizer.encode(word, add_special_tokens=False)

        # Remove the 'Ġ' token if present
        word = word[1:] if word.startswith('Ġ') else word

        # Find context embeddings for the first token of the word
        context_embeddings = roberta_model(torch.tensor([encoded_word])).last_hidden_state

        # Retrieve senses of the word from WordNet
        word_senses = get_word_senses(word)

        best_similarity = -1  # Initialize best similarity score
        best_sense = None     # Initialize best sense

        # Iterate over each sense of the word
        for sense in word_senses:
            # Obtain the embedding of the sense definition using RoBERTa model
            sense_embedding = roberta_model(torch.tensor([roberta_tokenizer.encode(sense.definition())])).last_hidden_state

            # Calculate cosine similarity between context embedding and sense embedding
            similarity = calculate_similarity(context_embeddings, sense_embedding)

            # Update best similarity and best sense if similarity is higher
            if similarity.mean() > best_similarity:
                best_similarity = similarity.mean()
                best_sense = sense
        # # Print the contextual embeddings
        print(f"Contextual embeddings for '{word}':")
        print(context_embeddings)
        print()
        # Print the best sense for the metaphoric word
        print(f"Metaphoric word: {word}")
        if best_sense is not None:
            print(f"Best sense: {best_sense.definition()}")
        else:
            print("No suitable sense found in WordNet")
        print()  # Add a newline for better readability

if __name__ == "__main__":
    while True:
        input_sentence = input("Enter a sentence (or 'quit' to exit): ")
        if input_sentence.lower() == 'quit':
            break
        perform_wsd(input_sentence)
