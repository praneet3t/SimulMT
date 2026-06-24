import os
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader


class SiMTDataset(Dataset):
    """
    PyTorch Dataset for English Ã¢â€ â€™ Telugu simultaneous translation.

    Each item returns a flat token sequence:
        [system prompt] [English source] [separator] [Telugu target] [EOS]

    along with exact byte offsets into that sequence so the masking module
    knows where English source and Telugu target begin and end.
    """

    def __init__(self, json_path, tokenizer, max_source_len=60, max_target_len=80):
        self.samples = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        self.tokenizer      = tokenizer
        self.max_source_len = max_source_len   # English source cap
        self.max_target_len = max_target_len   # Telugu target cap

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample      = self.samples[idx]
        source_text = sample["source"]   # English
        target_text = sample["target"]   # Telugu

        # ------------------------------------------------------------------
        # Build the flat string using Gemma 3's chat template.
        # apply_chat_template handles the token formatting (<start_of_turn> etc).
        # We only set the content; direction is controlled by the system message.
        #
        #   <bos><start_of_turn>system
        #   Translate the text below to Telugu.<end_of_turn>
        #   <start_of_turn>user
        #   {English sentence}<end_of_turn>
        #   <start_of_turn>model
        #   {Telugu translation}<end_of_turn><eos>
        # ------------------------------------------------------------------

        system_only_msgs = [
            {"role": "system", "content": "Translate the text below to Telugu."}
        ]
        prompt_only_str = self.tokenizer.apply_chat_template(
            system_only_msgs, tokenize=False, add_generation_prompt=False
        )

        with_source_msgs = [
            {"role": "system", "content": "Translate the text below to Telugu."},
            {"role": "user",   "content": source_text},
        ]
        prompt_source_str = self.tokenizer.apply_chat_template(
            with_source_msgs, tokenize=False, add_generation_prompt=False
        )
        prompt_source_sep_str = self.tokenizer.apply_chat_template(
            with_source_msgs, tokenize=False, add_generation_prompt=True
        )

        full_str = prompt_source_sep_str + target_text + self.tokenizer.eos_token

        # ------------------------------------------------------------------
        # Compute token offsets in the flat sequence
        # ------------------------------------------------------------------
        source_start = len(self.tokenizer.encode(prompt_only_str,        add_special_tokens=False))
        source_end   = len(self.tokenizer.encode(prompt_source_str,      add_special_tokens=False))
        target_start = len(self.tokenizer.encode(prompt_source_sep_str,  add_special_tokens=False))

        full_tokens = self.tokenizer.encode(full_str, add_special_tokens=False)
        total_len   = len(full_tokens)

        input_ids = torch.tensor(full_tokens, dtype=torch.long)
        labels    = input_ids.clone()
        labels[:target_start] = -100            # only supervise the Telugu target

        attention_mask = torch.ones(total_len, dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "source_start":   source_start,
            "source_end":     source_end,
            "target_start":   target_start,
            "pad_token_id":   self.tokenizer.pad_token_id,
            "source_text":    source_text,
            "target_text":    target_text,
        }


def collate_fn(batch):
    """
    Pad a list of samples to the longest sequence in the batch.
    - input_ids  Ã¢â€ â€™ pad with pad_token_id
    - labels     Ã¢â€ â€™ pad with -100
    - attention_mask Ã¢â€ â€™ pad with 0
    """
    max_len      = max(x["input_ids"].size(0) for x in batch)
    pad_token_id = batch[0].get("pad_token_id", 0)

    input_ids_list, attn_list, labels_list = [], [], []
    source_starts, source_ends, target_starts = [], [], []
    source_texts, target_texts = [], []

    for x in batch:
        curr = x["input_ids"].size(0)
        pad  = max_len - curr

        input_ids_list.append(torch.cat([x["input_ids"],
                                         torch.full((pad,), pad_token_id, dtype=torch.long)]))
        labels_list.append(torch.cat([x["labels"],
                                      torch.full((pad,), -100, dtype=torch.long)]))
        attn_list.append(torch.cat([x["attention_mask"],
                                    torch.zeros(pad, dtype=torch.long)]))

        source_starts.append(x["source_start"])
        source_ends.append(x["source_end"])
        target_starts.append(x["target_start"])
        source_texts.append(x["source_text"])
        target_texts.append(x["target_text"])

    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_list),
        "labels":         torch.stack(labels_list),
        "source_start":   torch.tensor(source_starts, dtype=torch.long),
        "source_end":     torch.tensor(source_ends,   dtype=torch.long),
        "target_start":   torch.tensor(target_starts, dtype=torch.long),
        "source_text":    source_texts,
        "target_text":    target_texts,
    }


def get_dataloaders(train_path, val_path, tokenizer, batch_size=4):
    train_ds = SiMTDataset(train_path, tokenizer)
    val_ds   = SiMTDataset(val_path,   tokenizer)

    cuda = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=cuda)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=cuda)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def run_tests():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-translate")

    train_path = os.path.join("simult_mt", "data", "filtered", "train.json")
    val_path   = os.path.join("simult_mt", "data", "filtered", "val.json")

    print("Creating DataLoader...")
    train_loader, val_loader = get_dataloaders(train_path, val_path, tokenizer, batch_size=4)

    print("Fetching first batch...")
    batch = next(iter(train_loader))

    print("\n=== Shapes ===")
    print(f"  input_ids:      {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels:         {batch['labels'].shape}")

    print("\n=== First Sample ===")
    input_ids    = batch["input_ids"][0]
    labels       = batch["labels"][0]
    source_start = batch["source_start"][0].item()
    source_end   = batch["source_end"][0].item()
    target_start = batch["target_start"][0].item()

    print(f"  source_start = {source_start}  (start of English source)")
    print(f"  source_end   = {source_end}    (end of English source)")
    print(f"  target_start = {target_start}  (start of Telugu target)")

    # Decoded label tokens should exactly match the raw Telugu target text
    label_tokens   = [t.item() for t in labels if t.item() != -100]
    decoded_labels = tokenizer.decode(label_tokens, skip_special_tokens=True).strip()
    expected       = batch["target_text"][0].strip()

    print(f"\n  Decoded labels:  '{decoded_labels[:80]}'")
    print(f"  Expected target: '{expected[:80]}'")

    if decoded_labels == expected:
        print("\nOVERALL TEST RESULT: PASS")
    else:
        print("\nOVERALL TEST RESULT: FAIL")
        print(f"  decoded:  {repr(decoded_labels)}")
        print(f"  expected: {repr(expected)}")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()


