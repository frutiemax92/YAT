
import spacy
import spacy.cli
spacy.cli.download("en_core_web_sm")

def compress_caption(caption,
                     class_label=None,
                     id_label=None,
                     reject_keywords=['image',
                                      'camera',
                                      'screen',
                                      'element', 
                                      'similar',
                                      'impression',
                                      'images',
                                      'such',
                                      'context',
                                      'setting',
                                      'appearance',
                                      'context']):
    # Load the spaCy model
    nlp = spacy.load('en_core_web_sm')

    # Process the text using spaCy
    doc = nlp(caption)

    result = []

    # Temporary variable to store multi-token named entities (e.g., "Bob Williams")
    proper_nouns = []

    # Extract multi-token named entities (e.g., "Bob Williams")
    for ent in doc.ents:
        if ent.label_ == 'PERSON' or ent.label_ == 'ORG':
            proper_nouns.append(ent.text.lower())  # Store as lowercase for consistency

    adjectives = []

    # Iterate through the tokens
    for token in doc:
        if token.pos_ == 'ADJ':  # If token is an adjective, add it to adjectives list
            adjectives.append(token.text.lower())

        elif token.pos_ == 'NOUN' or token.pos_ == 'PROPN':  # We include proper nouns and regular nouns
            # If the token is part of a named entity (e.g., "Bob Williams"), combine them
            found = False
            for proper_noun in proper_nouns:
                if token.text.lower() in proper_noun:
                    found = True
                    break
            
            if found:
                found = False
                for r in result:
                    if proper_noun in r:  # Only add once to the result
                        found = True
                        break
                if found == False:
                    if adjectives:
                        result.append(' '.join(adjectives) + ' ' + proper_noun)  # Title case for consistency
                        adjectives = []
                    else:
                        result.append(proper_noun)

            else:
                # Combine adjectives with the noun if there are preceding adjectives
                if adjectives:
                    result.append(' '.join(adjectives) + ' ' + token.text.lower())
                    adjectives = []  # Reset adjectives for the next noun
                else:
                    result.append(token.text.lower())

    def to_reject(r):
        return r not in reject_keywords

    # reject keywords
    result = filter(to_reject, result)

    # duplicates
    result = list(dict.fromkeys(result))

    if class_label == None or id_label == None:
        result_string = ''
        for r in result:
            result_string = result_string + r + ', '
        return result_string[:-2]

    for i in range(len(result)):
        result[i] = result[i].replace(class_label, id_label + ' ' + class_label)

    result_string = ''
    for r in result:
        result_string = result_string + r + ', '
    return result_string[:-2]

def remove_word(caption : str, word : str):
    caption = caption.replace(word, '')
    return caption