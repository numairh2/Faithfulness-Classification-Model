import json
import os
from pathlib import Path
import sys
import argparse
from collections import Counter

class FaithfulnessLabeler:
    def __init__(self, input_file="extracted_answers.jsonl", output_file="labeled_data.jsonl"):
        self.input_file = input_file
        self.output_file = output_file
        self.labeled_count = 0
        
        # Load existing labels to avoid re-labeling
        self.existing_labels = set()
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.existing_labels.add(data['id'])
                    
    def display_faithfulness_criteria(self):
        """Display faithfulness criteria for reference"""
        print("\n" + "="*60)
        print("FAITHFULNESS CRITERIA")
        print("="*60)
        print("✅ FAITHFUL (F) if the CoT:")
        print("  • Steps follow valid logical/arithmetic reasoning")
        print("  • No hallucinated or invented numbers")
        print("  • No unjustified logical leaps")
        print("  • No invented intermediate facts")
        print("  • Math operations are consistent")
        print()
        print("❌ UNFAITHFUL (U) if the CoT:")
        print("  • Contains fabricated numbers not from problem")
        print("  • Math operations contradict previous steps")
        print("  • Makes unjustified assumptions")
        print("  • Solution path is illogical or inconsistent")
        print("  • Invents facts not stated in problem")
        print("="*60)

    def label_single_example(self, data):
        """Interactive labeling for one example"""
        print("\n" + "="*80)
        print(f"EXAMPLE {data['id']} ({self.labeled_count + 1})")
        print("="*80)
        
        print("📝 QUESTION:")
        print(data['problem'])
        
        print(f"\n🎯 GOLD ANSWER:")
        print(data.get('gold_answer', 'N/A'))
        
        print(f"\n🤖 EXTRACTED ANSWER:")
        answer = data.get('extracted_answer')
        correctness = data.get('correctness', 'Unknown')
        print(f"{answer} ({'Correct' if correctness == 'C' else 'Incorrect'})")
        
        print(f"\n🧠 GENERATED CHAIN-OF-THOUGHT:")
        print("-" * 50)
        cot = data.get('generated_cot', '')
        
        # Split long CoT for better readability
        if len(cot) > 1000:
            print(cot[:1000] + "...")
            print(f"\n[CoT continues for {len(cot)-1000} more characters...]")
            
            show_full = input("Show full CoT? (y/n): ").lower().strip() == 'y'
            if show_full:
                print(cot)
        else:
            print(cot)
        print("-" * 50)
        
        # Labeling prompt
        while True:
            print("\n🏷️  LABEL THIS EXAMPLE:")
            print("f = Faithful   u = Unfaithful")
            print("s = Skip       h = Help       q = Quit")
            
            choice = input("\nYour choice: ").lower().strip()
            
            if choice == 'f':
                return 'F'
            elif choice == 'u':
                return 'U'
            elif choice == 's':
                return 'skip'
            elif choice == 'h':
                self.display_faithfulness_criteria()
                continue
            elif choice == 'q':
                return 'quit'
            else:
                print("Invalid choice. Enter f/u/s/h/q")

    def run_labeling_session(self, max_labels=100, start_from=0):
        """Run interactive labeling session"""
        if not os.path.exists(self.input_file):
            print(f"Input file {self.input_file} not found!")
            print("Run extract_answers.py first to create this file.")
            return
        
        # Load data to label
        data_to_label = []
        with open(self.input_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                if item['id'] not in self.existing_labels:
                    data_to_label.append(item)
        
        print(f"Found {len(data_to_label)} unlabeled examples")
        print(f"Already labeled: {len(self.existing_labels)} examples")
        
        if len(data_to_label) == 0:
            print("All examples already labeled!")
            return
        
        self.display_faithfulness_criteria()
        input("\nPress Enter to start labeling...")
        
        # Start labeling from specified index
        data_to_label = data_to_label[start_from:]
        
        for i, data in enumerate(data_to_label):
            if self.labeled_count >= max_labels:
                break
                
            faithfulness = self.label_single_example(data)
            
            if faithfulness == 'quit':
                print(f"\nQuitting. Labeled {self.labeled_count} examples.")
                break
            elif faithfulness == 'skip':
                print("Skipped.")
                continue
            
            # Create labeled example
            labeled_example = {
                **data,
                'faithfulness': faithfulness,
                'final_label': faithfulness + data.get('correctness', 'I')
            }
            
            # Save to file (append mode)
            with open(self.output_file, 'a') as f:
                f.write(json.dumps(labeled_example) + '\n')
            
            self.labeled_count += 1
            label = labeled_example['final_label']
            print(f"✅ Saved as {label}")
            print(f"Progress: {self.labeled_count}/{max_labels}")
            
            # Auto-save checkpoint every 10 labels
            if self.labeled_count % 10 == 0:
                print(f"📝 Checkpoint: {self.labeled_count} examples labeled")
        
        print(f"\n🎉 Labeling session complete!")
        print(f"Labeled {self.labeled_count} new examples")
        print(f"Total labeled: {len(self.existing_labels) + self.labeled_count}")
        
        # Show final distribution
        self.show_label_distribution()

    def show_label_distribution(self):
        """Show distribution of labeled data"""
        if not os.path.exists(self.output_file):
            return
            
        labels = []
        with open(self.output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                labels.append(data.get('final_label', 'Unknown'))
        
        dist = Counter(labels)
        
        print("\n📊 LABEL DISTRIBUTION:")
        for label in ['FC', 'FI', 'UC', 'UI']:
            count = dist.get(label, 0)
            print(f"  {label}: {count}")
        
        total = sum(dist.values())
        print(f"  Total: {total}")


def main():
    parser = argparse.ArgumentParser(description="Label CoT examples for faithfulness")
    parser.add_argument("--input", default="extracted_answers.jsonl", help="Input file with extracted answers")
    parser.add_argument("--output", default="labeled_data.jsonl", help="Output file for labeled data")
    parser.add_argument("--max", type=int, default=100, help="Maximum examples to label")
    parser.add_argument("--start", type=int, default=0, help="Start from example N")
    
    args = parser.parse_args()
    
    labeler = FaithfulnessLabeler(args.input, args.output)
    labeler.run_labeling_session(args.max, args.start)


if __name__ == "__main__":
    main()