# Fine-Tuned Customer Support Chatbot (LoRA + QLoRA)

## 📌 Overview
This project implements a **domain-specific customer support chatbot** by fine-tuning **open-source Large Language Models (LLMs)** such as **Mistral** and **Phi-2** using **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)** techniques.  
The chatbot is optimized for **low GPU memory environments** by leveraging **4-bit quantization**, enabling efficient fine-tuning on consumer-grade hardware without sacrificing accuracy.

---

## 🚀 Features
- **Domain-specific fine-tuning** of open-source LLMs.
- **LoRA + QLoRA** for parameter-efficient training.
- **4-bit quantization** for reduced memory footprint.
- Built using **entirely open-source libraries**.
- Runs on **low-end GPUs** (8GB VRAM capable).
- Easily deployable with **FastAPI** API endpoint.

---

## 🛠️ Tech Stack
| Category          | Tools & Technologies |
|-------------------|----------------------|
| LLM Framework     | [Hugging Face Transformers](https://huggingface.co/docs/transformers) |
| Fine-Tuning       | [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft) |
| Quantization      | [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) |
| Data Processing   | [Pandas](https://pandas.pydata.org/) |
| API Deployment    | [FastAPI](https://fastapi.tiangolo.com/) |
| Model Serving     | [Uvicorn](https://www.uvicorn.org/) |
| Tokenization      | Hugging Face Tokenizers |
| Environment Mgmt  | Python 3.10+ & virtualenv |

---

## 📂 Project Structure
customer-support-chatbot/
│
├── data/ # Training dataset (CSV/JSON)
│ └── customer_support_data.csv
│
├── model/ # Saved fine-tuned model
│
├── scripts/
│ ├── train.py # Fine-tuning script
│ ├── inference.py # Model inference script
│ └── api.py # FastAPI server for chatbot
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .env # API keys / environment variables (if needed)