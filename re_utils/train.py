import pdb
import json
from typing import Dict, List, Optional

import torch
from IPython import display
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import NerDataset
from datasets.embeddings_and_relations_dataset import EmbeddingsAndRelationsDataset
from datasets.ground_truth_relations_dataset import GroundTruthRelationsDataset
from models.bert_crf import BertCrf
from models.re_bert_crf import ReBertCrf
from re_utils.common import load_json
from re_utils.ner import get_tags_with_positions, get_mean_vector_from_segment
from focal_loss.focal_loss import FocalLoss


def dict_to_device(
        dict: Dict[str, torch.Tensor],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    for key, value in dict.items():
        dict[key] = value.to(device)
    return dict


def draw_plots(loss_history: List[float], f1: List[float], f1_relation: List[float] = []):
    display.clear_output(wait=True)

    f, (ax1, ax2, ax3) = plt.subplots(3)
    f.set_figwidth(15)
    f.set_figheight(10)

    
    ax1.set_title("training loss")
    ax2.set_title("f1 micro")
    ax3.set_title("f1 for class relation")

    ax1.plot(loss_history)
    ax2.plot(f1)
    ax3.plot(f1_relation)

    if not f1_relation:
        ax3.remove()

    plt.show()

    if len(loss_history) > 0:
        print(f"Current loss: {loss_history[-1]}")
    if len(f1) > 0:
        print(f"Current f1: {f1[-1]}")
    if len(f1_relation) > 0:
        print(f"Current f1_relation: {f1_relation[-1]}")


def train_ner(
        num_labels: int,
        bert_name: str,
        train_tokenized_texts_path: str,
        test_tokenized_texts_path: str,
        dropout: float,
        batch_size: int,
        epochs: int,
        log_every: int,
        lr_bert: float,
        lr_new_layers: float,
        use_crf: bool = True,
        save_to: Optional[str] = None,
        device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = BertCrf(num_labels, bert_name, dropout=dropout, use_crf=use_crf)
    model = model.to(device)
    model.train()

    train_dataset = NerDataset(train_tokenized_texts_path)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_function,
    )

    test_dataset = NerDataset(test_tokenized_texts_path)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_function,
    )

    optimizer = Adam(
        [
            {"params": model.start_transitions},
            {"params": model.end_transitions},
            {"params": model.hidden2label.parameters()},
            {"params": model.transitions},
            {"params": model.bert.parameters(), "lr": lr_bert},
        ],
        lr=lr_new_layers,
    )

    loss_history = []
    f1 = []

    step = 0

    for epoch in range(1, epochs + 1):
        for batch in tqdm(train_data_loader):
            step += 1

            optimizer.zero_grad()

            batch = dict_to_device(batch, device)
            # pdb.set_trace()

            try:
                loss = model(**batch)
            except:
                print(next(model.parameters()).requires_grad)

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if step % log_every == 0:
                model.eval()
                predictions = []
                ground_truth = []
                with torch.no_grad():
                    for batch in test_data_loader:
                        labels = batch["labels"].to(device)
                        del batch["labels"]
                        batch = dict_to_device(batch)

                        prediction = model.decode(**batch)

                        flatten_prediction = [item for sublist in prediction for item in sublist]
                        flatten_labels = torch.masked_select(labels, batch["attention_mask"].bool()).tolist()

                        predictions.extend(flatten_prediction)
                        ground_truth.extend(flatten_labels)
                f1_micro = f1_score(ground_truth, predictions, average="micro")
                f1.append(f1_micro)
                model.train()

            draw_plots(loss_history, f1)
            print(f"Epoch {epoch}/{epochs}")
    if save_to is not None:
        model.save_to(save_to)


def train_re(
        num_re_tags: int,
        batch_size: int,
        hidden_size: int,
        dropout: float,
        entity_tag_to_id_path: str,
        retag2id_path: str,
        re_train_data_path: str,
        re_test_data_path: str,
        test_relations_path: str,
        lr: float,
        epochs: int,
        log_every: int,
        test_deleted_relations_count: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    entity_tag_to_id = load_json(entity_tag_to_id_path)
    retag2id = load_json(retag2id_path)

    model = ReBertCrf(
        num_re_tags=num_re_tags, hidden_size=hidden_size, dropout=dropout, entity_tag_to_id=entity_tag_to_id
    )
    model = model.to(device)
    train_dataset = EmbeddingsAndRelationsDataset(re_data_path=re_train_data_path)
    test_dataset = EmbeddingsAndRelationsDataset(re_data_path=re_test_data_path)
    test_gt_relations_dataset = GroundTruthRelationsDataset(relations_path=test_relations_path)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_function)
    
    weights = torch.FloatTensor([1, 1/100]).to(torch.device(device))
    # criterion = CrossEntropyLoss(weight=weights)
    criterion = FocalLoss(gamma=12)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    f1_history = []
    f1_relation_history = []

    step = 0
    for epoch in range(1, epochs + 1):
        for batch in train_data_loader:
            step += 1
            optimizer.zero_grad()

            batch = dict_to_device(batch, device)
            relation_matrix_ground_truth = batch["relation_matrix"]

            logits = model(batch["seq_embedding"], batch["entities_embeddings"], batch["entities_tags"])

            m = torch.nn.Softmax(dim=-1)
            loss = criterion(m(logits.flatten(end_dim=2)), relation_matrix_ground_truth.flatten())
            #print(logits.flatten(end_dim=2), relation_matrix_ground_truth.flatten())
            loss.backward()

            loss_history.append(loss.item())

            optimizer.step()

            if step % log_every == 0:
                f1_micro, f1_relation = calc_f1_micro_and_f1_for_relation(
                    model,
                    test_dataset,
                    entity_tag_to_id,
                    retag2id,
                    test_gt_relations_dataset,
                    test_deleted_relations_count,
                    device,
                )
                # pdb.set_trace()
                f1_history.append(f1_micro)
                f1_relation_history.append(f1_relation)

            draw_plots(loss_history, f1_history, f1_relation_history)
            print(f"Epoch {epoch}/{epochs}")
    model.save_to('weights/re.pt')


def calc_f1_micro_and_f1_for_relation(
        re_model: ReBertCrf,
        test_re_dataset: EmbeddingsAndRelationsDataset,
        entity_tag_to_id: Dict[str, int],
        retag2id: Dict[str, int],
        gt_relations_dataset: Dataset,
        deleted_relations_count: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    re_model.eval()

    true_positive = 0
    false_positive = 0
    false_negative = deleted_relations_count
    true_positive_relation = 0
    false_negative_relation = 0
    false_positive_relation = 0
    # precision = 0
    # recall = 0

    for i, item in enumerate(test_re_dataset):
        seq_embedding = item["seq_embedding"]
        entities_embeddings = item["entities_embeddings"]
        entities_tags = item["entities_tags"]
        entities_positions = item["entities_positions"]
        gt_relation_matrix_on_ner = item["relation_matrix"]

        def prepare_tensor_for_model(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0).to(device)

        relation_matrix_pred = re_model(
            prepare_tensor_for_model(seq_embedding),
            prepare_tensor_for_model(entities_embeddings),
            prepare_tensor_for_model(entities_tags),
        )[0].argmax(dim=-1)
        gt_relation_matrix_on_ner = gt_relation_matrix_on_ner.to(device)

        # nayden = (relation_matrix_pred == 0).sum().item()

        # -------------------------- f1 для класса наличия отношения ------------------------------

        true_positive_relation += torch.sum((gt_relation_matrix_on_ner == 0) & (relation_matrix_pred == 0)).item()
        false_negative_relation += torch.sum((gt_relation_matrix_on_ner == 0) & (relation_matrix_pred != 0)).item()
        false_positive_relation += torch.sum((gt_relation_matrix_on_ner != 0) & (relation_matrix_pred == 0)).item()
        
        print(f'TP: {true_positive_relation}, FN: {false_negative_relation}, FP: {false_positive_relation}')
        precision_relation = true_positive_relation / (true_positive_relation + false_positive_relation) if (true_positive_relation + false_positive_relation) > 0 else 0
        recall_relation = true_positive_relation / (true_positive_relation + false_negative_relation) if (true_positive_relation + false_negative_relation) > 0 else 0
        
        f1_relation = 2 * precision_relation * recall_relation / (precision_relation + recall_relation) if (precision_relation + recall_relation) > 0 else 0
        
        # print(true_positive_relation+false_positive_relation, false_negative_relation+true_positive_relation, true_positive_relation)

        # -----------------------------------------------------------------------------------------

        # if nayden == 0:
        #     precision = 0
        # else:
        #     precision += true_ans / nayden

        
        # true_comm = (gt_relation_matrix_on_ner == 0).sum().item()
        # if true_comm == 0:
        #     recall = 0
        # else:
        #     recall += true_ans / true_comm

        # print(torch.unique(relation_matrix_pred, return_counts=True), torch.unique(gt_relation_matrix_on_ner, return_counts=True))

        no_relation_tag = len(retag2id)
        true_positive += torch.sum(
            relation_matrix_pred == gt_relation_matrix_on_ner
        ).item()

        false_positive += torch.sum(
            relation_matrix_pred != gt_relation_matrix_on_ner
        ).item()
        # no_relation_tag = len(retag2id)
        # true_positive += torch.sum(
        #     torch.logical_and(
        #         relation_matrix_pred == gt_relation_matrix_on_ner, gt_relation_matrix_on_ner != no_relation_tag
        #     )
        # ).item()
        # false_positive += torch.sum(
        #     torch.logical_and(
        #         relation_matrix_pred != gt_relation_matrix_on_ner,
        #         gt_relation_matrix_on_ner != no_relation_tag,
        #     )
        # ).item()

        entities_tags_list = entities_tags.tolist()
        entities_positions_list = entities_positions.tolist()

        entity_pos_to_id = {(pos[0], pos[1]): id for id, pos in enumerate(entities_positions_list)}
        gt_relations = gt_relations_dataset[i]

        # pdb.set_trace()

        for relation in gt_relations:
            arg1_tag = relation["arg1_tag"]
            arg2_tag = relation["arg2_tag"]
            arg1_pos = tuple(relation["arg1_pos"])
            arg2_pos = tuple(relation["arg2_pos"])

            if arg1_pos not in entity_pos_to_id or arg2_pos not in entity_pos_to_id:
                false_negative += 1
                continue

            entity1_id = entity_pos_to_id[arg1_pos]
            entity2_id = entity_pos_to_id[arg2_pos]

            if (
                    entity_tag_to_id[arg1_tag] != entities_tags_list[entity1_id]
                    or entity_tag_to_id[arg2_tag] != entities_tags_list[entity2_id]
            ):
                false_negative += 1
                continue

            if relation_matrix_pred[entity1_id][entity2_id].item() != relation["tag"]:
                false_negative += 1

    re_model.train()
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    # betta = 2
    if (precision + recall) == 0:
        f1_micro = 0
    else:
        f1_micro = 2 * (precision * recall) / (precision + recall)
    # f1_micro = true_positive / (true_positive + 0.5 * (false_positive + false_negative))
    # f_betta = (1+betta*betta)*precision*recall/((precision+recall)*(betta*betta))
    return f1_micro, f1_relation
