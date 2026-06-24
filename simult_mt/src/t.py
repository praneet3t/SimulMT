from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained( "sarvamai/sarvam-translate", device_map="cpu")

print(model)