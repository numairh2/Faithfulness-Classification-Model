import json
import os
import random
from pathlib import Path
from collections import Counter
import numpy as np
import argparse

def load_processed_data(input_file="data_processed/fcm_processed.jsonl"):
    """Load processed FCM data"""
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found!")
        print("Run prepare_fcm_data.py first.")
        return None
    
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    return data

def stratified_split(data, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Create stratified train/dev/test splits"""
    
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Split ratios must sum to 1.0")
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Group by label
    label_groups = {}
    for item in data:
        label = item['label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    # Shuffle each group
    for label in label_groups:
        random.shuffle(label_groups[label])
    
    train_data = []
    dev_data = []
    test_data = []
    
    # Split each label group
    for label, items in label_groups.items():
        n = len(items)
        
        train_end = int(n * train_ratio)
        dev_end = train_end + int(n * dev_ratio)
        
        train_data.extend(items[:train_end])
        dev_data.extend(items[train_end:dev_end])
        test_data.extend(items[dev_end:])
        
        print(f"Label {label}: {len(items)} total -> Train: {train_end}, Dev: {dev_end-train_end}, Test: {len(items)-dev_end}")
    
    # Final shuffle
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    
    return train_data, dev_data, test_data

def upsample_minority_classes(data, target_ratio=0.15):
    """Upsample minority classes to reduce imbalance"""
    
    # Count classes
    label_counts = Counter(item['label'] for item in data)
    total_samples = len(data)
    
    print("Original class distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/total_samples*100:.1f}%)")
    
    # Find target count (based on largest class)
    max_count = max(label_counts.values())
    target_count = max(int(max_count * target_ratio), 10)  # At least 10 samples
    
    # Group by label
    label_groups = {}
    for item in data:
        label = item['label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    upsampled_data = []
    
    for label, items in label_groups.items():
        if len(items) < target_count:
            # Upsample by repeating samples
            multiplier = target_count // len(items)
            remainder = target_count % len(items)
            
            upsampled_items = items * multiplier
            if remainder > 0:
                upsampled_items.extend(random.sample(items, remainder))
            
            print(f"Upsampled {label}: {len(items)} -> {len(upsampled_items)}")
            upsampled_data.extend(upsampled_items)
        else:
            upsampled_data.extend(items)
    
    random.shuffle(upsampled_data)
    
    print(f"\nAfter upsampling: {len(data)} -> {len(upsampled_data)} examples")
    
    return upsampled_data

def save_split(data, split_name, output_dir):
    """Save a data split to file"""
    output_file = f"{output_dir}/fcm_{split_name}.jsonl"
    
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(data)} examples to {output_file}")
    return output_file

def analyze_splits(train_data, dev_data, test_data):
    """Analyze and report on the splits"""
    
    def analyze_split(data, split_name):
        label_dist = Counter(item['label'] for item in data)
        cot_lengths = [item['metadata']['cot_length'] for item in data]
        
        return {
            'name': split_name,
            'size': len(data),
            'label_distribution': dict(label_dist),
            'avg_cot_length': np.mean(cot_lengths),
            'std_cot_length': np.std(cot_lengths)
        }
    
    train_stats = analyze_split(train_data, 'train')
    dev_stats = analyze_split(dev_data, 'dev')
    test_stats = analyze_split(test_data, 'test')
    
    print("\n" + "="*60)
    print("SPLIT ANALYSIS")
    print("="*60)
    
    for stats in [train_stats, dev_stats, test_stats]:
        print(f"\n{stats['name'].upper()} SET:")
        print(f"  Size: {stats['size']}")
        print(f"  Label distribution:")
        for label in ['FC', 'FI', 'UC', 'UI']:
            count = stats['label_distribution'].get(label, 0)
            pct = count / stats['size'] * 100 if stats['size'] > 0 else 0
            print(f"    {label}: {count} ({pct:.1f}%)")
        print(f"  Avg CoT length: {stats['avg_cot_length']:.1f} ± {stats['std_cot_length']:.1f}")
    
    return {
        'train': train_stats,
        'dev': dev_stats,
        'test': test_stats
    }

def create_dataset_splits(input_file="data_processed/fcm_data.jsonl",
                         output_dir="data_processed",
                         upsample=True,
                         train_ratio=0.7,
                         dev_ratio=0.15,
                         test_ratio=0.15,
                         random_seed=42):
    """Create train/dev/test splits with optional upsampling"""
    
    # Load data
    data = load_processed_data(input_file)
    if not data:
        return
    
    print(f"Loaded {len(data)} examples")
    
    # Optional upsampling
    if upsample:
        print("\nApplying upsampling for class balance...")
        data = upsample_minority_classes(data)
    
    # Create splits
    print(f"\nCreating splits ({train_ratio:.0%}/{dev_ratio:.0%}/{test_ratio:.0%})...")
    train_data, dev_data, test_data = stratified_split(
        data, train_ratio, dev_ratio, test_ratio, random_seed
    )
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = save_split(train_data, 'train', output_dir)
    dev_file = save_split(dev_data, 'dev', output_dir)
    test_file = save_split(test_data, 'test', output_dir)
    
    # Analyze splits
    split_stats = analyze_splits(train_data, dev_data, test_data)
    
    # Save analysis
    analysis_file = f"{output_dir}/split_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump({
            'split_config': {
                'train_ratio': train_ratio,
                'dev_ratio': dev_ratio,
                'test_ratio': test_ratio,
                'random_seed': random_seed,
                'upsampling_applied': upsample
            },
            'statistics': split_stats
        }, f, indent=2)
    
    print(f"\n✅ Dataset splits created successfully!")
    print(f"📈 Analysis saved to {analysis_file}")
    
    return {
        'train': train_data,
        'dev': dev_data,
        'test': test_data,
        'files': {
            'train': train_file,
            'dev': dev_file,
            'test': test_file,
            'analysis': analysis_file
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Create dataset splits for FCM training")
    parser.add_argument("--input", default="data_processed/fcm_data.jsonl",
                       help="Input processed data file")
    parser.add_argument("--output-dir", default="data_processed",
                       help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training set ratio")
    parser.add_argument("--dev-ratio", type=float, default=0.15,
                       help="Development set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Test set ratio")
    parser.add_argument("--no-upsample", action="store_true",
                       help="Disable upsampling for class balance")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    create_dataset_splits(
        input_file=args.input,
        output_dir=args.output_dir,
        upsample=not args.no_upsample,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()