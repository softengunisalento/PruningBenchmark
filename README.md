# README COPIATO E CHE DEVO MODIFICARE

# Benchmarking Pruning Techniques to Reduce the Energy Impact of Large Language Models: the Case of _LLMPruner_

This repo investigates the impact of LLM model pruning techniques, specifically LLMPruner, on both model efficiency and environmental sustainability. The research analyzes a selection of open-source pre-trained LLM models from huggingface. The results showed that an average reduction of 20\% in parameters can result in energy savings on the order of 10\%, but with an average accuracy loss of 12\%, which is not negligible and very model and task dependent. In some specific scenarios the approach proved more effective, but overall the benefits were not substantial enough to make the method immediately applicable on a large scale without further optimization.

---

## Usage

Clone the repository

```bash
git clone https://github.com/Cyber-Vadok/GreenPruning/
```

Create a venv and install dependencies:

```bash
cd GreenPruning/
pip install -r requirements.txt
```

Here's a simple example of applying PTQ to a PyTorch model:

```python
if args.quantization_bits == 4:
  model = AutoModelForMaskedLM.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,
    device_map=device
  )
elif args.quantization_bits == 8:
  model = AutoModelForMaskedLM.from_pretrained(
    MODEL_ID,
    load_in_8bit=True,
    device_map=device
  )
else:
  raise ValueError("Only 4-bit and 8-bit quantization are supported.")
```

---

## Performance Benchmarks

The evaluation of model quantization highlights significant improvements in model efficiency and environmental impact, especially in terms of size reduction, inference speed, and energy consumption, though not without trade-offs in accuracy.

### Test Environment

* **CPU**: AMD Ryzen 9 7900X
* **GPU**: NVIDIA GeForce RTX 4090

---

## Results Summary

### Accuracy

<a href="./assets/my-table.svg">
  <img src="./assets/accuracy.svg" alt="Accuracy Table" width="1000"/>
</a>

### Energy Consumption

<a href="./assets/my-table.svg">
  <img src="./assets/energy.svg" alt="Accuracy Table" width="1000"/>
</a>

## Miglioramento torch-pruning
Aggiunta parte per ignorare i layer iniziali e finali
```
    layer_names = [name for name, _ in model.named_modules() if name.endswith(("self_attn", "mlp"))]
    # *2 = self_attn + mlp
    start_idx = 0
    end_idx = len(layer_names)
    if args.ignore_first_x_layers > 0:
        start_idx = args.ignore_first_x_layers * 2
    if args.ignore_last_x_layers > 0:
        end_idx -= args.ignore_last_x_layers * 2
    layers_to_process = layer_names[start_idx:end_idx]
```

## Miglioramento lm-eval-harness
Aggiunto Codecarbon

```
    tracker.start()
        print("Running", reqtype, "requests")
        resps = getattr(lm, reqtype)([req.args for req in reqs])
        
        resps = [
            x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
        ]

        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append((i, resp))

    tracker.stop()
```

## Credits
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
It includes code from:
 * [Torch-Pruning](https://github.com/VainF/Torch-Pruning) (MIT License)
 * [LLM-Pruner](https://github.com/horseee/LLM-Pruner) (Apache 2.0 License)
 * [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (MIT License)

