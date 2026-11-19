"""Question/paragraph cross-encoder and validated training data.

Importing this module does not download a model or start training.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class RelevancePair:
    qid: str
    question: str
    paragraph: str  # Includes title, matching the original notebook.
    label: int

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip()
               for v in (self.qid, self.question, self.paragraph)):
            raise ValueError('qid, question, and paragraph must be nonempty strings')
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError('label must be integer 0 or 1')


def load_pairs(path):
    """Read JSONL exported from train_df/dev_df; ignore extra DataFrame fields."""
    pairs = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                pairs.append(RelevancePair(**{k: row[k] for k in ('qid', 'question', 'paragraph', 'label')}))
    if not pairs:
        raise ValueError('Pair file is empty')
    validate_groups(pairs)
    return pairs


def validate_groups(pairs):
    questions, seen = {}, set()
    for pair in pairs:
        if pair.qid in questions and questions[pair.qid] != pair.question:
            raise ValueError('One qid has inconsistent question text')
        questions[pair.qid] = pair.question
        key = (pair.qid, pair.paragraph)
        if key in seen:
            raise ValueError('Duplicate question/paragraph pair')
        seen.add(key)


def validate_splits(train, dev):
    validate_groups(train)
    validate_groups(dev)
    if not train or not dev:
        raise ValueError('Train and dev must both be nonempty')
    if {p.qid for p in train} & {p.qid for p in dev}:
        raise ValueError('Train/dev question IDs overlap')
    for name, pairs in [('train', train), ('dev', dev)]:
        if {p.label for p in pairs} != {0, 1}:
            raise ValueError(f'{name} must contain positive and negative examples')
        grouped = defaultdict(list)
        for p in pairs:
            grouped[p.qid].append(p)
        if any(not any(p.label for p in group) for group in grouped.values()):
            raise ValueError(f'Every {name} question needs annotated positive evidence')


def add_verified_hard_negatives(train, path):
    """Accept only manually verified negatives belonging to training questions.

    Never infer a negative label from the presence of 'not' or a changed date.
    Reviewers must judge usefulness as evidence for the exact question.
    """
    questions = {p.qid: p.question for p in train}
    seen = {(p.qid, p.paragraph) for p in train}
    result, review_records = list(train), []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            pair = RelevancePair(**{k: row[k] for k in ('qid', 'question', 'paragraph', 'label')})
            if pair.qid not in questions or questions[pair.qid] != pair.question:
                raise ValueError('Hard negative must match an existing training question')
            if pair.label != 0 or row.get('verified') is not True:
                raise ValueError('Hard negatives need label=0 and verified=true')
            for field in ('reason', 'reviewer', 'category'):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f'Hard negative needs a nonempty {field}')
            key = (pair.qid, pair.paragraph)
            if key in seen:
                raise ValueError('Hard negative duplicates an existing pair; do not relabel gold evidence')
            seen.add(key)
            result.append(pair)
            review_records.append(row)
    if not review_records:
        raise ValueError('Hard-negative file is empty')
    return result, review_records


def relevance_metrics(pairs, scores, k=2):
    """Paragraph metrics; all-gold@k means every annotated positive was selected."""
    if len(pairs) != len(scores) or not pairs or k < 1:
        raise ValueError('Need nonempty pairs, matching scores, and positive k')
    if not all(isfinite(float(s)) for s in scores):
        raise ValueError('Scores must be finite')
    tp = fp = fn = correct = 0
    grouped = defaultdict(list)
    for i, (p, score) in enumerate(zip(pairs, scores)):
        pred = float(score) >= 0
        tp += pred and p.label == 1
        fp += pred and p.label == 0
        fn += not pred and p.label == 1
        correct += pred == bool(p.label)
        grouped[p.qid].append(i)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    all_kept, recalls = [], []
    for indices in grouped.values():
        gold = {i for i in indices if pairs[i].label}
        if not gold:
            raise ValueError('Ranking evaluation requires positive evidence for every question')
        kept = set(sorted(indices, key=lambda i: (-float(scores[i]), i))[:k])
        all_kept.append(gold <= kept)
        recalls.append(len(gold & kept) / len(gold))
    return dict(accuracy=correct / len(pairs), precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                all_gold_at_k=sum(all_kept) / len(all_kept),
                gold_paragraph_recall_at_k=sum(recalls) / len(recalls), k=k)


class RelevanceClassifier:
    """Wrap a one-logit Transformer; inject model/tokenizer for offline testing."""

    def __init__(self, model, tokenizer, max_length=256, device='cpu'):
        import torch
        if type(max_length) is not int or max_length < 8:
            raise ValueError('max_length must be at least 8')
        if model.config.num_labels != 1:
            raise ValueError('Expected a binary classifier with one output logit')
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @classmethod
    def from_pretrained(cls, path='cross-encoder/ms-marco-MiniLM-L-6-v2',
                        max_length=None, device='cpu'):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        metadata = Path(path) / 'relevance_config.json'
        config = json.loads(metadata.read_text()) if metadata.is_file() else {}
        return cls(AutoModelForSequenceClassification.from_pretrained(path),
                   AutoTokenizer.from_pretrained(path),
                   max_length=max_length if max_length is not None else config.get('max_length', 256),
                   device=device)

    def encode(self, questions, paragraphs):
        return self.tokenizer(list(questions), list(paragraphs), padding=True,
                              truncation='only_second', max_length=self.max_length,
                              return_tensors='pt').to(self.device)

    def score(self, question, paragraphs, batch_size=32):
        """Inference accepts no gold labels; returns raw logits in input order."""
        import torch
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if not isinstance(question, str) or not question.strip():
            raise ValueError('question must be nonempty text')
        if isinstance(paragraphs, str):
            raise ValueError('paragraphs must be a sequence of paragraphs, not one string')
        if any(not isinstance(p, str) or not p.strip() for p in paragraphs):
            raise ValueError('paragraphs must contain nonempty text')
        self.model.eval()
        scores = []
        with torch.inference_mode():
            for start in range(0, len(paragraphs), batch_size):
                chunk = paragraphs[start:start + batch_size]
                enc = self.encode([question] * len(chunk), chunk)
                scores.extend(self.model(**enc).logits.reshape(-1).float().cpu().tolist())
        return scores

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / 'relevance_config.json').write_text(json.dumps({'max_length': self.max_length}, indent=2))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    seed: int = 42
    selection_k: int = 2

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.epochs, self.batch_size, self.selection_k)):
            raise ValueError('epochs, batch_size, and selection_k must be positive')
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if not isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError('warmup_fraction must be in [0,1)')


def evaluate(classifier, pairs, batch_size=32, k=2):
    import torch
    classifier.model.eval()
    scores, loss_sum = [], 0.0
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            enc = classifier.encode([p.question for p in chunk], [p.paragraph for p in chunk])
            logits = classifier.model(**enc).logits.reshape(-1).float()
            labels = torch.tensor([p.label for p in chunk], dtype=torch.float32, device=classifier.device)
            loss_sum += torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='sum').item()
            scores.extend(logits.cpu().tolist())
    return {**relevance_metrics(pairs, scores, k), 'loss': loss_sum / len(pairs)}


def fit(classifier, train, dev, output_dir, config=TrainConfig()):
    """Explicit training entry point. Choose best epoch on dev, never test.

    Refuse an existing output directory; never delete a teammate's checkpoint.
    Load output_dir/best after training for the chosen model, not the last epoch.
    """
    import math
    import random
    import torch
    validate_splits(train, dev)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if classifier.device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)
    positives = sum(p.label for p in train)
    positive_weight = (len(train) - positives) / positives
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=classifier.device))
    optimizer = torch.optim.AdamW(classifier.model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(train) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    warmup = max(1, int(total_steps * config.warmup_fraction)) if config.warmup_fraction else 0
    rng = random.Random(config.seed)
    history, best_key, step = [], None, 0
    for epoch in range(config.epochs):
        classifier.model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        total_loss = 0.0
        for start in range(0, len(train), config.batch_size):
            chunk = [train[i] for i in order[start:start + config.batch_size]]
            # Set LR explicitly before each update: linear warm-up, then decay.
            factor = ((step + 1) / warmup if step < warmup else
                      (total_steps - step) / max(1, total_steps - warmup))
            for group in optimizer.param_groups:
                group['lr'] = config.learning_rate * factor
            optimizer.zero_grad(set_to_none=True)
            enc = classifier.encode([p.question for p in chunk], [p.paragraph for p in chunk])
            labels = torch.tensor([p.label for p in chunk], dtype=torch.float32, device=classifier.device)
            logits = classifier.model(**enc).logits.reshape(-1).float()
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; stop and inspect the inputs')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(chunk)
            step += 1
        metrics = evaluate(classifier, dev, config.batch_size, config.selection_k)
        record = {'epoch': epoch + 1, 'train_loss': total_loss / len(train), 'dev': metrics}
        history.append(record)
        key = (metrics['all_gold_at_k'], -metrics['loss'])
        if best_key is None or key > best_key:
            best_key = key
            classifier.save(output_dir / 'best')
            (output_dir / 'best_epoch.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
        (output_dir / 'training_report.json').write_text(json.dumps({
            'config': asdict(config), 'max_length': classifier.max_length,
            'positive_weight': positive_weight, 'train_pairs': len(train), 'dev_pairs': len(dev),
            'history': history}, indent=2))
    return history
