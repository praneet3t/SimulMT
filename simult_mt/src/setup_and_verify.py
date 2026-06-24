import os
import sys
import traceback
import torch

def main():
    configs_dir = os.path.join("simult_mt", "configs")
    os.makedirs(configs_dir, exist_ok=True)
    
    # 1. Verify CUDA
    print("=== 1. Verifying CUDA ===")
    cuda_available = torch.cuda.is_available()
    gpu_info_path = os.path.join(configs_dir, "gpu_info.txt")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) # GB
        gpu_info = f"CUDA Available: True\nGPU Name: {gpu_name}\nTotal VRAM: {total_vram:.2f} GB"
    else:
        gpu_info = "CUDA Available: False\nGPU Name: N/A (CPU only or non-CUDA GPU)\nTotal VRAM: N/A"
    
    print(gpu_info)
    with open(gpu_info_path, "w", encoding="utf-8") as f:
        f.write(gpu_info + "\n")

    # 2. Load the model and tokenizer
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        model_name = "sarvamai/sarvam-translate"
        print(f"\n=== 2. Loading tokenizer and model: {model_name} ===")
        
        # 4-bit NF4 Config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading model in 4-bit NF4 quantization...")
        # device_map="auto" maps layers to CUDA if available, but will error if CUDA is missing
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        print("Model loaded successfully!")
        
        # 3. Model architecture
        print("\n=== 3. Saving model architecture ===")
        arch_path = os.path.join(configs_dir, "model_architecture.txt")
        with open(arch_path, "w", encoding="utf-8") as f:
            for name, module in model.named_modules():
                f.write(f"{name}: {module.__class__.__name__}\n")
        print(f"Model architecture saved to {arch_path}")
        
        # 4. Filter attention layers
        print("\n=== 4. Saving attention layers ===")
        att_layers_path = os.path.join(configs_dir, "attention_layers.txt")
        target_substrings = ["q_proj", "k_proj", "v_proj", "o_proj"]
        att_layers = []
        for name, module in model.named_modules():
            if any(sub in name for sub in target_substrings):
                att_layers.append(name)
        
        with open(att_layers_path, "w", encoding="utf-8") as f:
            for layer in att_layers:
                f.write(layer + "\n")
        print(f"Found {len(att_layers)} target attention layers. Saved to {att_layers_path}")
        
        # 5. Run English sentences through the model (English -> Telugu)
        print("\n=== 5. Running translation ===")
        sentences = [
            "Floods have arrived in the state.",
            "He is reading a book.",
            "She is cooking."
        ]
        
        tgt_lang = "Telugu"
        translations = []
        for sentence in sentences:
            messages = [
                {"role": "system", "content": f"Translate the text below to {tgt_lang}."},
                {"role": "user", "content": sentence}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Tokenize & push to model's device
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            # Greedy Decode
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            # Extract generated tokens
            input_len = inputs.input_ids.shape[1]
            output_ids = outputs[0][input_len:]
            translated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            translations.append((sentence, translated_text))
            print(f"Source: {sentence}\nTranslation: {translated_text}\n")
            
        trans_path = os.path.join(configs_dir, "translation_output.txt")
        with open(trans_path, "w", encoding="utf-8") as f:
            for src, tgt in translations:
                f.write(f"Source: {src}\nTranslation: {tgt}\n\n")
        print(f"Translations saved to {trans_path}")
        
    except Exception as e:
        print("\n!!! Error encountered during model load or execution !!!")
        tb = traceback.format_exc()
        print(tb)
        
        err_path = os.path.join(configs_dir, "error_traceback.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(tb)
        print(f"Full traceback saved to {err_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()


