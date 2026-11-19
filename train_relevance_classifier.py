"""Explicit baseline/proposed training command; --validate-only loads no model."""
import argparse
import hashlib
import json
from pathlib import Path

from relevance_model import (RelevanceClassifier, TrainConfig, add_verified_hard_negatives,
                             fit, load_pairs, validate_splits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-pairs', required=True)
    parser.add_argument('--dev-pairs', required=True)
    parser.add_argument('--hard-negatives', help='Optional reviewed training-only JSONL')
    parser.add_argument('--output-dir', help='New checkpoint directory, required for actual training')
    parser.add_argument('--base-model', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.max_length < 8:
        parser.error('--max-length must be at least 8')
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                         learning_rate=args.learning_rate, seed=args.seed)
    train, dev = load_pairs(args.train_pairs), load_pairs(args.dev_pairs)
    validate_splits(train, dev)
    reviews = []
    if args.hard_negatives:
        train, reviews = add_verified_hard_negatives(train, args.hard_negatives)
    validate_splits(train, dev)
    print(f'Validated {len(train)} training / {len(dev)} dev pairs; {len(reviews)} reviewed hard negatives')
    if args.validate_only:
        return
    if not args.output_dir or Path(args.output_dir).exists():
        parser.error('Actual training requires --output-dir pointing to a NEW directory')
    import torch
    # Set seed before loading a model in case any parameters require initialization.
    torch.manual_seed(args.seed)
    classifier = RelevanceClassifier.from_pretrained(args.base_model, args.max_length, args.device)
    fit(classifier, train, dev, args.output_dir, config)
    provenance = {'base_model': args.base_model, 'device': args.device,
                  'torch_version': torch.__version__, 'hard_negative_count': len(reviews)}
    for name in ('train_pairs', 'dev_pairs', 'hard_negatives'):
        value = getattr(args, name)
        if value:
            provenance[name] = {'path': value, 'sha256': hashlib.sha256(Path(value).read_bytes()).hexdigest()}
    (Path(args.output_dir) / 'input_provenance.json').write_text(json.dumps(provenance, indent=2))
    if reviews:
        (Path(args.output_dir) / 'hard_negative_reviews.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in reviews))


if __name__ == '__main__':
    main()
