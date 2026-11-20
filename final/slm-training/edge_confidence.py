"""Day 3 helper for Lab 20: how sure is the model? (the probability of each intent)

Idea: after the text  {"intent": "  the model chooses the next tokens of the label. For each of the nine labels we ask the model
"how likely is this label here?", add up the log-probabilities of its tokens, and turn the nine scores into nine probabilities
that add up to 1. A model that is sure gives one label a high probability. A model that is guessing spreads it out.
"""
import math

import torch

import edge_lab as el
import edge_nlu


def label_probabilities(tok, model, system, user_text):
    """Returns a dictionary: label -> probability (the nine probabilities add up to 1)."""
    prompt = el.make_prompt_text(tok, system, user_text) + '{"intent": "'
    prefix_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    label_ids_list = []
    sequences = []
    for label in edge_nlu.INTENTS:
        label_ids = tok(label, add_special_tokens=False)["input_ids"]
        label_ids_list.append(label_ids)
        sequences.append(prefix_ids + label_ids)
    longest = 0
    for sequence in sequences:
        if len(sequence) > longest:
            longest = len(sequence)
    input_ids = []
    attention = []
    for sequence in sequences:
        padding = longest - len(sequence)
        input_ids.append(sequence + [tok.pad_token_id] * padding)
        attention.append([1] * len(sequence) + [0] * padding)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor(input_ids).to(model.device), attention_mask=torch.tensor(attention).to(model.device)).logits
    scores = []
    for index in range(len(label_ids_list)):
        total = 0.0
        for position in range(len(label_ids_list[index])):
            row = logits[index, len(prefix_ids) + position - 1].float()
            total = total + float(torch.log_softmax(row, dim=-1)[label_ids_list[index][position]])
        scores.append(total)
    biggest = max(scores)
    weights = []
    for score in scores:
        weights.append(math.exp(score - biggest))
    weight_sum = sum(weights)
    probabilities = {}
    for index in range(len(edge_nlu.INTENTS)):
        probabilities[edge_nlu.INTENTS[index]] = weights[index] / weight_sum
    return probabilities


def best_two(probabilities):
    """Returns (best_label, best_probability, second_label, second_probability)."""
    first_label = ""
    first = -1.0
    second_label = ""
    second = -1.0
    for label in probabilities:
        value = probabilities[label]
        if value > first:
            second_label, second = first_label, first
            first_label, first = label, value
        elif value > second:
            second_label, second = label, value
    return first_label, first, second_label, second


def show_confidence(tok, model, system, texts, threshold=0.9):
    """For each message: the best intent and its probability, the runner-up, and whether the assistant would ask back."""
    for text in texts:
        if text.strip() == "":
            continue                                   # an empty row is skipped
        probabilities = label_probabilities(tok, model, system, text)
        best, best_p, second, second_p = best_two(probabilities)
        if best_p >= threshold:
            decision = "ANSWER"
        else:
            decision = "ASK BACK"
        print(text)
        print("    best:", best, round(best_p, 2), "| runner-up:", second, round(second_p, 2), "|", decision)
