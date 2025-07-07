from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, LlamaTokenizer

app = FastAPI()

model_path = "fine_tuning"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = LlamaTokenizer.from_pretrained(model_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

device_target = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map={"": device_target},
    trust_remote_code=True,
    torch_dtype=torch.float16
)

if torch.cuda.is_available():
    model.to("cuda")

class InputText(BaseModel):
    prompt: str
    max_length: int = 100

@app.get("/")
def root():
    return {"message": "fine-tune llm API is running!"}

@app.post("/generate")
def generate_text(data: InputText):
    eval_prompt = f"""Answer the following question **first**. You may provide more relevant facts after the answer, but always start by directly answering the question.

Q: {data.prompt}
A:"""

    inputs = tokenizer(eval_prompt, return_tensors="pt").to(device_target)

    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n Model's Answer:\n", answer)

    torch.cuda.empty_cache()

    return {"response": answer}
