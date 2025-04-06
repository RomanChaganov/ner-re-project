import json
import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm
from transformers import LayoutXLMModel, AutoTokenizer

from re_utils.common import load_jsonl
from re_utils.ner import get_tags_with_positions, get_mean_vector_from_segment


def configure_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--layoutxlm-name", type=str, default="microsoft/layoutxlm-base", help="LayoutXLM model name")
    parser.add_argument("--labeled-texts", type=str, default="resources/data/train/labeled_texts.jsonl")
    parser.add_argument("--relations", type=str, default="resources/data/train/relations.jsonl")
    parser.add_argument("--label2id", type=str, default="resources/data/train/label2id.json")
    parser.add_argument("--retag2id", type=str, default="resources/data/train/retag2id.json")
    return parser


def main(args: Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dir = os.path.dirname(args.labeled_texts)

    model = LayoutXLMModel.from_pretrained(args.layoutxlm_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.layoutxlm_name)
    model.eval()

    labeled_texts = load_jsonl(args.labeled_texts)
    relations = load_jsonl(args.relations)

    label2id = json.load(open(args.label2id))
    id2label = {id: label for label, id in label2id.items()}

    # entity tags
    entity_tags_set = set()
    for label in label2id:
        if label != "O":
            entity_tags_set.add(label.split("-")[1])
    entity_tag_to_id = {tag: id for id, tag in enumerate(entity_tags_set)}

    retag2id = json.load(open(args.retag2id))
    no_relation_tag = len(retag2id)

    with open(os.path.join(dir, "re_data.jsonl"), "w") as out:
        for labeled_text, text_relations in tqdm(zip(labeled_texts, relations)):
            assert labeled_text["id"] == text_relations["id"]

            input_ids = torch.tensor([labeled_text["input_ids"]], device=device)
            attention_mask = torch.ones_like(input_ids, device=device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[0]  # seq_len x hidden_dim

            full_seq_embedding = embeddings.mean(dim=0).tolist()

            labels = labeled_text["labels"]  # без CRF теперь
            tags_pos = get_tags_with_positions(labels, id2label)

            relation_matrix = np.full((len(tags_pos), len(tags_pos)), fill_value=no_relation_tag)

            for i, first_arg in enumerate(tags_pos):
                for j, second_arg in enumerate(tags_pos):
                    for relation in text_relations["relations"]:
                        if (
                                relation["arg1_tag"] == first_arg["tag"] and
                                relation["arg2_tag"] == second_arg["tag"] and
                                relation["arg1_pos"] == first_arg["pos"] and
                                relation["arg2_pos"] == second_arg["pos"]
                        ):
                            relation_matrix[i][j] = relation["tag"]

            entities_positions = [item["pos"] for item in tags_pos]
            entities_embeddings = [
                get_mean_vector_from_segment(embeddings, pos[0], pos[1]).tolist() for pos in entities_positions
            ]
            entities_tags = [entity_tag_to_id[item["tag"]] for item in tags_pos]

            json.dump({
                "id": labeled_text["id"],
                "seq_embedding": full_seq_embedding,
                "entities_embeddings": entities_embeddings,
                "relation_matrix": relation_matrix.tolist(),
                "entities_tags": entities_tags,
                "entities_positions": entities_positions,
            }, out)
            out.write("\n")

    json.dump(entity_tag_to_id, open(os.path.join(dir, "entity_tag_to_id.json"), "w"))


if __name__ == "__main__":
    main(configure_arg_parser().parse_args())
