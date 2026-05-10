# models folder

Put converted `.gguf` models here if you want Ollama to run a fully fine-tuned exported model.

Example:

```text
models/java-dto-qwen2.5-coder-0.5b.Q4_K_M.gguf
```

Then edit:

```text
ollama/Modelfile.custom-gguf-template
```

and run:

```bash
ollama create java-dto-finetuned -f ollama/Modelfile.custom-gguf-template
```
