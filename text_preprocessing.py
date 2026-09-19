"""Training-compatible preprocessing, copied from Experiment 2 for standalone deployment."""

import re
import html

CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bcant\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bwont\b": "will not",
    r"\bn't\b": " not",
    r"\bain't\b": "is not",
    r"\bdon't\b": "do not",
    r"\bdont\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdoesnt\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bdidnt\b": "did not",
    r"\bisn't\b": "is not",
    r"\bisnt\b": "is not",
    r"\baren't\b": "are not",
    r"\barent\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bwasnt\b": "was not",
    r"\bweren't\b": "were not",
    r"\bwerent\b": "were not",
    r"\bhaven't\b": "have not",
    r"\bhavent\b": "have not",
    r"\bhasn't\b": "has not",
    r"\bhasnt\b": "has not",
    r"\bhadn't\b": "had not",
    r"\bhadnt\b": "had not",
    r"\bshouldn't\b": "should not",
    r"\bwouldn't\b": "would not",
    r"\bcouldn't\b": "could not",
    r"\bi'm\b": "i am",
    r"\bi've\b": "i have",
    r"\bi'll\b": "i will",
    r"\bi'd\b": "i would",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bwhat's\b": "what is",
    r"\bthere's\b": "there is",
    r"\blet's\b": "let us"
}

NEGATION_TRIGGERS = {
    "not", "no", "never", "cannot", "cant", "n't", "neither", "nor", 
    "without", "hardly", "scarcely", "barely", "rarely", "seldom", 
    "lack", "lacking", "nowhere", "nothing", "none"
}

CLAUSE_DELIMITERS = {
    ".", ",", "!", "?", ";", ":", "-", "--", "(", ")", "[", "]", "{", "}",
    "but", "however", "although", "though", "yet", "except", "while", "nevertheless",
    "instead", "that", "which", "who", "whom", "whose", "because", "since",
    "unless", "whereas", "wherever", "after", "before", "so", "and", "or"
}

REMOVAL_VERBS = {
    "shake", "stop", "eliminate", "dispel", "prevent", "avoid", "contain",
    "suppress", "quell", "overcome", "control", "hide", "resist", "forget",
    "lose", "calm"
}

def expand_contractions(text: str) -> str:
    """Expands common English contractions for consistent negation recognition."""
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    for pattern, replacement in CONTRACTIONS.items():
        text_lower = re.sub(pattern, replacement, text_lower)
    return text_lower

def apply_negation_tagging(text: str, max_window: int = 3) -> str:
    """
    Applies bounded negation scope tagging.
    Appends '_NEG' to tokens within max_window words following a negation trigger,
    terminating immediately upon encountering clause delimiters, subordinating conjunctions,
    relative pronouns, or removal/cessation verbs.
    
    Example:
      "I was not disappointed with the outcome, but I was nervous"
      --> "i was not disappointed_NEG with_NEG the_NEG outcome , but i was nervous"
      
      "I could not shake the fear that everything might disappear"
      --> "i could not shake_NEG the fear that everything might disappear"
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    expanded = expand_contractions(text)
    tokens = re.findall(r"\w+|[^\w\s]", expanded, re.UNICODE)
    
    tagged_tokens = []
    scope_remaining = 0
    
    for token in tokens:
        token_lower = token.lower()
        
        # Punctuation or clause delimiter immediately clears negation scope
        if token_lower in CLAUSE_DELIMITERS or re.match(r"^[.,!?;:\-â€“â€”]$", token):
            scope_remaining = 0
            tagged_tokens.append(token)
            continue
            
        # Negation trigger activates scope for max_window words
        if token_lower in NEGATION_TRIGGERS:
            scope_remaining = max_window
            tagged_tokens.append(token)
            continue
            
        if scope_remaining > 0:
            # If the token is a removal verb (e.g., "shake" in "could not shake")
            if token_lower in REMOVAL_VERBS:
                tagged_tokens.append(f"{token}_NEG")
                # Terminate negation scope because the object being shaken/avoided is retained!
                scope_remaining = 0
                continue
                
            if token.isalnum() and not token.isdigit():
                tagged_tokens.append(f"{token}_NEG")
            else:
                tagged_tokens.append(token)
                
            scope_remaining -= 1
        else:
            tagged_tokens.append(token)
            
    return " ".join(tagged_tokens)

def clean_tweet_text(text: str) -> str:
    """
    Cleans raw tweet text for sentiment analysis:
    1. HTML unescaping (&amp; -> &, &lt; -> <, etc.)
    2. Removes URLs (http/https)
    3. Removes Twitter user mentions (@username)
    4. Normalizes whitespaces and strips leading/trailing spaces
    """
    if not isinstance(text, str):
        return ""
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove @ mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove special control characters but keep punctuation and emojis
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # Normalize multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
