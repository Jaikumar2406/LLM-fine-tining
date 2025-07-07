# 🦙 LLaMA-2 7B Chat Fine-Tuned on Hawaii Dataset 🌺

This project demonstrates the complete fine-tuning, backend deployment, and frontend integration of Meta’s LLaMA-2 7B Chat model on a custom "Hawaii" dataset using **LoRA (Low-Rank Adaptation)**.

---

## 📌 Features

- 🔍 Fine-tuned `meta-llama/Llama-2-7b-chat-hf` with a custom Hawaii text dataset
- ⚡ Used **PEFT** and **BitsAndBytes** for efficient 4-bit fine-tuning
- 🚀 Built a RESTful **FastAPI** backend for text generation
- 🌐 Created a responsive **HTML/CSS/JavaScript** frontend chat interface
- 📉 Achieved loss reduction from `2.87 → 0.99` showing successful model convergence
- 🔁 Fully working end-to-end pipeline from input prompt to LLM-generated response

---

## 🧠 Technologies Used

| Layer       | Tools/Frameworks                             |
|-------------|----------------------------------------------|
| Model       | `LLaMA-2 7B`, `transformers`, `peft`, `bnb`  |
| Backend     | `FastAPI`, `PyTorch`        |
| Frontend    | `HTML`, `CSS`, `JS`                  |
| Optimization| `LoRA`, `4-bit Quantization (nf4)`           |

---
![Screenshot 2025-07-08 002853](https://github.com/user-attachments/assets/025290d7-f459-49f2-919b-ebd0d929efb6)
![Screenshot 2025-07-08 002906](https://github.com/user-attachments/assets/4f6a4d73-7a1f-408d-90b4-1cadb360ec10)
![Screenshot 2025-07-08 002920](https://github.com/user-attachments/assets/629566e9-c35b-40cd-956d-c4adff678ae7)


## 🗃️ Dataset

The model was fine-tuned on a custom plain text file containing information and facts about **Hawaii**.

Example data:
Hawaii is the only U.S. state located in the Pacific Ocean.
It consists of 137 islands spread over 1,500 miles.
The capital of Hawaii is Honolulu.
---

## 📊 Fine-Tuning Logs (Sample)

| Step | Loss    |
|------|---------|
| 10   | 2.8791  |
| 100  | 2.7545  |
| 200  | 2.5627  |
| 300  | 1.7836  |
| 400  | 1.7688  |
| 500  | 0.9948  |
| 550  | 1.0671  |

✅ Model shows consistent convergence — ready for generation tasks!

---

## 🧪 API Usage

**Endpoint:** `POST /generate`
{
  "prompt": "What is the capital of Hawaii?",
  "max_length": 500
}
**Response:**
{
  "response": "The capital of Hawaii is Honolulu."
}
