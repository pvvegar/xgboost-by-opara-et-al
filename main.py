from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
import statistics
import string
import time
import nltk
import pandas as pd
from typing import TypedDict
from textstat import textstat
from xgboost import XGBClassifier

# Setup NLTK Resources
def setup_nltk():
    required_resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    for resource_id, resource_path in required_resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"NLTK resource '{resource_id}' not found. Downloading...")
            nltk.download(resource_id, quiet=True)
            print(f"'{resource_id}' downloaded successfully.")

setup_nltk()


# Metrics
def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

SCORING = {
    "F1": f1_score,
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "MCC": matthews_corrcoef,
    "ROC AUC": roc_auc_score,
    "PRC AREA": average_precision_score,
    "FPR": false_positive_rate,
}

def save_scores(experiment: str, index: str, values: dict) -> None:
    columns = list(SCORING.keys()) + ["training_time", "inference_time"]
    scores = pd.DataFrame(columns=columns)
    row = {}
    for metric in SCORING:
        if metric in values:
            val = values[metric]
            row[metric] = round(val, 4)
    row["training_time"] = round(values.get("training_time", 0), 4)
    row["inference_time"] = round(values.get("inference_time", 0), 4)
    scores.loc[index] = row
    print(scores)

# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# TypedDict specifying the structure and types for all extracted text features.
class FeaturesDict(TypedDict):
    
    # Lexical features
    word_count: int
    char_count: int  # including letters, digits, punctuation and spaces
    avg_word_length: float
    sentence_count: int  # count of sentences based on punctuation (".", "!", "?")
    avg_sentence_length: float
    unique_word_count: int
    lexical_diversity: float  # unique words / total words
    number_of_emails: int  # count of @
    uppercase_word_count: int
    uppercase_ratio: float  # uppercase words / total words
    complex_word_count: int  # count of words with more than 6 characters
    avg_syllables_per_word: float
    
    # Syntactic features
    comma_count: int
    semicolon_count: int
    colon_count: int
    exclamation_count: int
    quotation_count: int
    dash_count: int
    sentence_complexity: (
        float  # count of Literal["and", "or", "but", "because"] / sentence_count
    )
    clause_density: float  # sentence_complexity / sentence_count
    pronoun_density: float  # count of pronouns / word_count
    preposition_density: (
        float  # count of Literal["in", "on", "at", "by", "with"] / word_count
    )
    function_word_density: (
        float  # count of Literal["the", "is", "at", "which", "on"] / word_count
    )
    dot_frequency: float  # count of "." / char_count
    comma_frequency: float  # count of "," / char_count
    exclamation_frequency: float  # count of "!" / char_count
    double_dot_frequency: float  # count of ":" / char_count
    dash_frequency: float  # count of "-" / char_count
    quotation_frequency: float  # count of """ / char_count
    left_parentheses_frequency: float  # count of "(" / char_count
    right_parentheses_frequency: float  # count of ")" / char_count
    slash_frequency: float  # count of "/" / char_count
    backslash_frequency: float  # count of "\" / char_count
    punctuation_variety: int  # unique punctuation marks
    
    # Readability scores
    flesch_reading_ease: float
    smog_index: float
    dale_chall_readability_score: float
    coleman_liau_index: float
    gunning_fog_index: float
    
    # Word category features
    number_of_pronouns: (
        int  # count of Literal["I", "you", "he", "she", "it", "we", "they"]
    )
    first_person_pronoun_count: int  # count of "I", "we"
    second_person_pronoun_count: int  # count of "you"
    imperative_verb_count: (
        int  # count of Literal["click", "verify", "submit", "download", "update"]
    )
    modal_verb_count: int  # count of Literal["can", "could", "should"]
    uncertainty_adverb_count: int  # count of Literal["maybe", "perhaps", "possibly"]
    technical_jargon_count: (
        int  # count of Literal["security", "account", "update", "technical"]
    )
    promotional_word_count: int  # count of Literal["free", "offer", "deal"]
    
    # Email-Specific features
    number_of_attachments_mentions: int  # count of "attachment" or "attached"
    
    # Complexity features
    bigram_count: int  # count of bigrams
    trigram_count: int  # count of trigrams
    word_lenght_variation: float  # standard deviation of word lengths
    
    # Stylistic features
    politeness_markers_count: int  # count of Literal["please", "thank", "appreciate"]
    aggressiveness_markers_count: int  # count of Literal["must", "now", "immediately"]
    urgency_markers_count: int  # count of Literal["urgent", "immediately", "asap"]
    conditional_phrases_count: int  # count of Literal["if", "unless"]
    personalisation_markers_count: (
        int  # count of Literal["you", "your"] or name mentions
    )

# Word lists based on the paper's descriptions in Table 1
LINKING_CONJUNCTIONS: set[str] = {"and", "or", "but", "because"}
PREPOSITIONS: set[str] = {"in", "on", "at", "by", "with"}
FUNCTION_WORDS: set[str] = {"the", "is", "at", "which", "on"}
PERSONAL_PRONOUNS: set[str] = {"i", "you", "he", "she", "it", "we", "they"}
FIRST_PERSON_PRONOUNS: set[str] = {"i", "we"}
SECOND_PERSON_PRONOUNS: set[str] = {"you"}
IMPERATIVE_VERBS: set[str] = {"click", "verify", "submit", "download", "update"}
MODAL_VERBS: set[str] = {"can", "could", "should"}
UNCERTAINTY_ADVERBS: set[str] = {"maybe", "perhaps", "possibly"}
TECHNICAL_JARGON: set[str] = {"security", "account", "update", "technical"}
PROMOTIONAL_WORDS: set[str] = {"free", "offer", "deal"}
POLITENESS_MARKERS: set[str] = {"please", "thank", "appreciate"}
AGGRESSIVENESS_MARKERS: set[str] = {"must", "now", "immediately"}
URGENCY_MARKERS: set[str] = {"urgent", "immediately", "asap"}
CONDITIONAL_PHRASES: set[str] = {"if", "unless"}
PERSONALISATION_MARKERS: set[str] = {"you", "your"}

# Function to extract features from a given text
def extract_features(text: str) -> FeaturesDict:
    if not text.strip():
        return dict.fromkeys(FeaturesDict.__annotations__, 0)

    # Tokenize words and sentences
    words = nltk.word_tokenize(text)
    lower_words = [word.lower() for word in words]
    sentences = nltk.sent_tokenize(text)

    # Basic counts to avoid recalculation and handle division by zero
    char_count = len(text)
    word_count = len(words)
    sentence_count = len(sentences)

    # Lexical Features 
    unique_word_count = len(set(lower_words))
    lexical_diversity = unique_word_count / word_count if word_count > 0 else 0
    uppercase_word_count = sum(1 for word in words if word.isupper() and len(word) > 1)
    complex_word_count = sum(1 for word in words if len(word) > 6)

    # Syntactic Features
    comma_count = text.count(",")
    semicolon_count = text.count(";")
    colon_count = text.count(":")
    exclamation_count = text.count("!")
    quotation_count = text.count('"') + text.count("'")
    dash_count = text.count("-")

    linking_conj_count = sum(word in LINKING_CONJUNCTIONS for word in lower_words)
    sentence_complexity = (
        linking_conj_count / sentence_count if sentence_count > 0 else 0
    )

    clause_density = sentence_complexity / sentence_count if sentence_count > 0 else 0

    pronoun_count = sum(word in PERSONAL_PRONOUNS for word in lower_words)
    preposition_count = sum(word in PREPOSITIONS for word in lower_words)
    function_word_count = sum(word in FUNCTION_WORDS for word in lower_words)

    # Complexity Features 
    word_lengths = [len(word) for word in words]
    word_lenght_variation = (
        statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0
    )

    # Final Dictionary Assembly 
    features: FeaturesDict = {

        # Lexical Features
        "word_count": word_count,
        "char_count": char_count,
        "avg_word_length": sum(word_lengths) / word_count if word_count > 0 else 0,
        "sentence_count": sentence_count,
        "avg_sentence_length": word_count / sentence_count if sentence_count > 0 else 0,
        "unique_word_count": unique_word_count,
        "lexical_diversity": lexical_diversity,
        "number_of_emails": text.count("@"),
        "uppercase_word_count": uppercase_word_count,
        "uppercase_ratio": uppercase_word_count / word_count if word_count > 0 else 0,
        "complex_word_count": complex_word_count,
        "avg_syllables_per_word": textstat.syllable_count(text) / word_count
        if word_count > 0
        else 0,

        # Syntactic Features
        "comma_count": comma_count,
        "semicolon_count": semicolon_count,
        "colon_count": colon_count,
        "exclamation_count": exclamation_count,
        "quotation_count": quotation_count,
        "dash_count": dash_count,
        "sentence_complexity": sentence_complexity,
        "clause_density": clause_density,
        "pronoun_density": pronoun_count / word_count if word_count > 0 else 0,
        "preposition_density": preposition_count / word_count if word_count > 0 else 0,
        "function_word_density": function_word_count / word_count
        if word_count > 0
        else 0,
        "dot_frequency": text.count(".") / char_count if char_count > 0 else 0,
        "comma_frequency": comma_count / char_count if char_count > 0 else 0,
        "exclamation_frequency": exclamation_count / char_count
        if char_count > 0
        else 0,
        "double_dot_frequency": colon_count / char_count if char_count > 0 else 0,
        "dash_frequency": dash_count / char_count if char_count > 0 else 0,
        "quotation_frequency": quotation_count / char_count if char_count > 0 else 0,
        "left_parentheses_frequency": text.count("(") / char_count
        if char_count > 0
        else 0,
        "right_parentheses_frequency": text.count(")") / char_count
        if char_count > 0
        else 0,
        "slash_frequency": text.count("/") / char_count if char_count > 0 else 0,
        "backslash_frequency": text.count("\\") / char_count if char_count > 0 else 0,
        "punctuation_variety": len(
            {char for char in text if char in string.punctuation}
        ),

        # Readability Scores
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "smog_index": textstat.smog_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "gunning_fog_index": textstat.gunning_fog(text),

        # Word Category Features
        "number_of_pronouns": pronoun_count,
        "first_person_pronoun_count": sum(
            word in FIRST_PERSON_PRONOUNS for word in lower_words
        ),
        "second_person_pronoun_count": sum(
            word in SECOND_PERSON_PRONOUNS for word in lower_words
        ),
        "imperative_verb_count": sum(word in IMPERATIVE_VERBS for word in lower_words),
        "modal_verb_count": sum(word in MODAL_VERBS for word in lower_words),
        "uncertainty_adverb_count": sum(
            word in UNCERTAINTY_ADVERBS for word in lower_words
        ),
        "technical_jargon_count": sum(word in TECHNICAL_JARGON for word in lower_words),
        "promotional_word_count": sum(
            word in PROMOTIONAL_WORDS for word in lower_words
        ),

        # Email-Specific Features
        "number_of_attachments_mentions": lower_words.count("attachment"),

        # Complexity Features
        "bigram_count": len(list(nltk.ngrams(words, 2))),
        "trigram_count": len(list(nltk.ngrams(words, 3))),
        "word_lenght_variation": word_lenght_variation,

        # Stylistic Features
        "politeness_markers_count": sum(
            word in POLITENESS_MARKERS for word in lower_words
        ),
        "aggressiveness_markers_count": sum(
            word in AGGRESSIVENESS_MARKERS for word in lower_words
        ),
        "urgency_markers_count": sum(word in URGENCY_MARKERS for word in lower_words),
        "conditional_phrases_count": sum(
            word in CONDITIONAL_PHRASES for word in lower_words
        ),
        "personalisation_markers_count": sum(
            word in PERSONALISATION_MARKERS for word in lower_words
        ),
    }
    return features


X_train = pd.DataFrame(train_df["text"].apply(extract_features).tolist())
X_test = pd.DataFrame(test_df["text"].apply(extract_features).tolist())
y_train = train_df["label"].values
y_test = test_df["label"].values

print("Feature extraction complete.")

train_begin = time.time()
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=420)
model.fit(X_train, y_train)

train_time = time.time() - train_begin
start_infer = time.time()
xgb_probs = model.predict_proba(X_test)[:, 1]
xgb_preds = (xgb_probs > 0.5).astype(int)
infer_time = time.time() - start_infer

save_scores(
    "XGBoost",
    "xgb",
    {
        metric: SCORING[metric](
            y_test,
            xgb_preds if "ROC" not in metric and "PRC" not in metric else xgb_probs,
        )
        for metric in SCORING
    }
    | {"training_time": train_time, "inference_time": infer_time},
)