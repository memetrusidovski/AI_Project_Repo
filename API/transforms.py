from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForTableQuestionAnswering

tokenizer2 = AutoTokenizer.from_pretrained(
    "google/tapas-mini-finetuned-wtq")
model2 = TFAutoModelForTableQuestionAnswering.from_pretrained(
    "google/tapas-mini-finetuned-wtq")

data = {
    "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
    "Age": ["56", "45", "59"],
    "Number of movies": ["87", "53", "34"],
    "Context": ["Albania, on Southeastern Europe’s Balkan Peninsula, is a small country with Adriatic and Ionian coastlines and an interior crossed by the Albanian Alps. The country has many castles and archaeological sites. Capital Tirana centers on sprawling Skanderbeg Square, site of the National History Museum, with exhibits spanning antiquity to post-communism, and frescoed Et’hem Bey Mosque",
                "Albert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. Einstein is best known for developing the theory of relativity, but he also made important contributions to the development of the theory of quantum mechanics.",
                "Apple Inc. is an American multinational technology company headquartered in Cupertino, California, United States."],
}

table = pd.DataFrame.from_dict(data)
queries = ["Where is albania?",
           "Who's 56 years old?", "How old is Brad Pitt?"]

inputs = tokenizer2(table=table, queries=queries,
                    padding="max_length", return_tensors="tf")

outputs = model2(**inputs)

logits = outputs.logits
logits_aggregation = outputs.logits_aggregation

s = tokenizer2.convert_logits_to_predictions(
    inputs, logits, logits_agg=logits_aggregation, cell_classification_threshold=0.5)


