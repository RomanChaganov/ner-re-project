import glob
import json
import torch
from tqdm import tqdm

import numpy as np
from collections import defaultdict

from transformers import AutoTokenizer
from nltk.tokenize import WordPunctTokenizer

from models.bert_crf import BertCrf
from models.re_bert_crf import ReBertCrf
from re_utils.ner import get_tags_with_positions, get_mean_vector_from_segment



class Tag:
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos
        
    def __repr__(self):
        return f'Tag(name={self.name}, pos={self.pos})'


DEBUG = True
NUM_LABELS = 5
DROPOUT = 0
USE_CRF = True

NUM_RE_TAGS = 2
HIDDEN_SIZE = 768

BERT_NAME = "ai-forever/ruBert-base"
BERT_CRF_PATH = "weights/bert-crf.pt"
RE_BERT = "weights/re.pt"
LABEL2ID = "resources/data/train/label2id.json"
RETAG2ID = "resources/data/train/retag2id.json"

entities_positions = None
tags_pos = None


def tokenize(text, tokenizer, nltk_tokenizer=WordPunctTokenizer(), max_length=512):
    tokenized_text_spans = list(nltk_tokenizer.span_tokenize(text))
    words = [text[span[0] : span[1]] for span in tokenized_text_spans]
    encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False, max_length=max_length, truncation=True, padding='max_length')
    input_ids = encoded["input_ids"]
    words_ids_for_tokens = encoded.word_ids()
    
    return input_ids, words_ids_for_tokens, words
    
    
def ner_out(input_ids, model, id2label, entity_tag_to_id, device):
    global entities_positions, tags_pos
    attention_mask = torch.ones(1, len(input_ids), device=device)
    input_ids = torch.tensor([input_ids], device=device)
    
    _, batched_bert_embeddings = model.get_bert_features(input_ids, attention_mask)
    bert_embeddings = batched_bert_embeddings[0]
    full_seq_embedding = get_mean_vector_from_segment(bert_embeddings, 0, len(bert_embeddings))
    labels = model.decode(input_ids, attention_mask)[0]
    
    tags_pos = get_tags_with_positions(labels, id2label)
    entities_positions = [item["pos"] for item in tags_pos]
    entities_embeddings = torch.tensor([
        get_mean_vector_from_segment(bert_embeddings, pos[0], pos[1]).tolist() for pos in entities_positions
    ], dtype=torch.float)
    
    entities_tags = torch.tensor([entity_tag_to_id[item["tag"]] for item in tags_pos], dtype=torch.long)
    
    return full_seq_embedding, entities_embeddings, entities_tags
    
    
def re_out(entities_description, model, device):
    entities_description = map(lambda x: x.unsqueeze(0).to(device), entities_description)    
    seq_embedding, entities_embeddings, entities_tags = entities_description
    
    relation_matrix_pred = model(seq_embedding, entities_embeddings, entities_tags)
    relation_matrix_pred = relation_matrix_pred[0].argmax(dim=-1)
    
    relations = []
    
    for i in range(len(relation_matrix_pred)):
        for j in range(len(relation_matrix_pred)):
            if relation_matrix_pred[i][j] == 0: # to do no_relation_tag
                tag1 = Tag(tags_pos[i]['tag'], entities_positions[i])
                tag2 = Tag(tags_pos[j]['tag'], entities_positions[j])
                relations.append((tag1, tag2))  
    
    return relation_matrix_pred, relations

    
def run(text):
    device = "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)  
    nltk_tokenizer = WordPunctTokenizer()
    
    input_ids, words_ids_for_tokens, words = tokenize(text, tokenizer, nltk_tokenizer)
    
    with open(LABEL2ID, "r") as label2id_file:
        label2id = json.load(label2id_file)
    id2label = {id: label for label, id in label2id.items()}
    
    with open(RETAG2ID, "r") as retag2id_file:
        retag2id = json.load(retag2id_file)
    no_relation_tag = len(retag2id)
    
    entity_tags_set = set()
    for label, id in label2id.items():
        if label == "O":
            continue
        entity_tags_set.add(label.split("-")[1])
    entity_tag_to_id = {tag: id for id, tag in enumerate(entity_tags_set)}
    
    model = BertCrf(NUM_LABELS, BERT_NAME, DROPOUT, USE_CRF)
    model.load_from(BERT_CRF_PATH)
    model = model.to(device)
    model.eval()
    
    entities_description = ner_out(input_ids, model, id2label, entity_tag_to_id, device)
    
    re_model = ReBertCrf(NUM_RE_TAGS, HIDDEN_SIZE, DROPOUT, entity_tag_to_id)
    re_model.load_from(RE_BERT)
    re_model = re_model.to(device)
    re_model.eval()
    
    relation_matrix_pred, relations = re_out(entities_description, re_model, device)
    
    i = 0
    for tag1, tag2 in relations:
        if tag1.name != 'KEY':
            continue
            
        start_pos1, end_pos1 = tag1.pos
        start_pos2, end_pos2 = tag2.pos
        
        words1 = set(words_ids_for_tokens[start_pos1:end_pos1])
        words2 = set(words_ids_for_tokens[start_pos2:end_pos2])
        
        print(f"{tag1.name}: {' '.join([words[i] for i in words1])}")
        print(f"{tag2.name}: {' '.join([words[i] for i in words2])}")
        print()
        
        i += 1
    
    print(f'Найдено {i} связей')
    
    print("Sexassfully!!")


if __name__ == '__main__':
    with open('text.txt', 'r', encoding='UTF8') as f:
        text = f.read()
        
    run(text)
