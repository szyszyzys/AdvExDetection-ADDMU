import textattack
import transformers
from textattack import Attack
from textattack import Attacker
from textattack.datasets import HuggingFaceDataset

dataset = HuggingFaceDataset("imdb", split="test")
from textattack.search_methods import GreedyWordSwapWIR
from textattack.goal_functions import UntargetedClassification
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance

from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019

def build(model_wrapper):
    transformation = WordSwapEmbedding(max_candidates=100)

    stopwords = set(
        ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone",
         "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow",
         "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been",
         "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but",
         "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don",
         "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone",
         "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn",
         "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon",
         "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into",
         "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me",
         "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't",
         "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody",
         "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only",
         "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s",
         "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something",
         "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them",
         "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these",
         "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under",
         "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't",
         "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
         "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose",
         "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd",
         "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
    )

    input_column_modification = InputColumnModification(
        ["premise", "hypothesis"], {"premise"}
    )

    use_constraint = UniversalSentenceEncoder(
        threshold=0.840845057,
        metric="angular",
        compare_against_original=False,
        window_size=15,
        skip_text_shorter_than_window=True,
    )
    constraints = [RepeatModification(),
                   StopwordModification(stopwords=stopwords),
                   input_column_modification,
                   WordEmbeddingDistance(min_cos_sim=0.5),
                   PartOfSpeech(allow_verb_noun_swap=True),
                   use_constraint
                   ]

    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model_wrapper, target_max_score=70)
    #
    # Greedily swap words with "Word Importance Ranking".
    #
    search_method = GreedyWordSwapWIR(wir_method="delete")

    return Attack(goal_function, constraints, transformation, search_method)


# Load target model, tokenizer, and model_wrapper
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

attack_args = textattack.AttackArgs(
    num_examples=10,
    log_to_csv="log.csv",
    checkpoint_interval=5,
    checkpoint_dir="checkpoints",
    disable_stdout=False
)
dataset = HuggingFaceDataset("imdb", split="test")

# Now, let's make the attack from the 4 components:
# attack = build(model_wrapper)
attack = TextFoolerJin2019.build(model_wrapper)
# attack = Attack(goal_function, constraints, transformation, search_method)
attacker = Attacker(attack, dataset, attack_args)
res = attacker.attack_dataset()


print('============================')
print(res)