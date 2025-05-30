# README COPIATO E CHE DEVO MODIFICARE

# Benchmarking pruning techniques to reduce the energy impact of LLM models

This repo investigates the impact of LLM model pruning techniques, specifically LLMPruner, on both model efficiency and environmental sustainability. The research analyzes a selection of open-source pre-trained LLM models from huggingface. The results showed that an average reduction of 20\% in parameters can result in energy savings on the order of 15\%, but with an average accuracy loss of 12\%, which is not negligible and very model- and task-dependent. In some specific scenarios the approach proved more effective, but overall the benefits were not substantial enough to make the method immediately applicable on a large scale without further optimization.

---

## Usage

Clone the repository and install manually:

```bash
git clone https://github.com/Cyber-Vadok/GreenPruning/
cd quantization-toolkit
```

Create a venv and than:

```bash
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

## Supported Frameworks

* PyTorch
* TensorFlow

---

## Quantization Methods Implemented

* **Post-Training Quantization (PTQ)**

  * Dynamic quantization
  * Per-layer quantization

---

## Performance Benchmarks

The evaluation of model quantization highlights significant improvements in model efficiency and environmental impact, especially in terms of size reduction, inference speed, and energy consumption, though not without trade-offs in accuracy.

### Test Environment

* **CPU**: AMD Ryzen 9 7900X
* **GPU**: NVIDIA GeForce RTX 4090
* **Frameworks**: PyTorch 3.13

---

## Results Summary

### Feature Extraction and Image Classification

Quantization delivers strong results for **feature extraction tasks**, such as sentence embedding. For example, `BAAI/bge-small-en-v1.5` exhibits only an 8% increase in MSE while achieving an 80% reduction in model size. Similarly, in **image classification** tasks (e.g., `microsoft/resnet-50`), quantization achieves a 75% memory footprint reduction with minimal impact on accuracy, as summarized in Table \[tab\:fea\_iclass].

| Model                  | Tasks | INT8 mean MSE | INT4 mean MSE | INT8 Size | INT4 Size |
| ---------------------- | ----- | ------------- | ------------- | --------- | --------- |
| BAAI/bge-base-en-v1.5  | FE    | 4.48e-7       | 1.54e-6       | 25.00%    | 15.30%    |
| BAAI/bge-small-en-v1.5 | FE    | 7.80e-7       | 1.64e-6       | 25.00%    | 17.04%    |
| microsoft/resnet-50    | IC    | 7.16e-8       | 9.60e-9       | 25.00%    | 25.00%    |

### Sentence Transformers and Energy Impact

In **sentence similarity** tasks, INT8 quantization offers clear environmental advantages. For several models, energy consumption is reduced by up to 50% with negligible loss in accuracy. `all-mpnet-base-v2`, for example, reduces energy usage to 92.78% while maintaining strong accuracy (Table \[tab\:sen\_tra]).

| Model             | INT8 mean MSE | INT4 mean MSE | INT8 Energy % | INT4 Energy % |
| ----------------- | ------------- | ------------- | ------------- | ------------- |
| all-MiniLM-L6-v2  | 1.18e-7       | 8.43e-7       | 100.06%       | 111.10%       |
| all-mpnet-base-v2 | 3.09e-7       | 2.17e-7       | 92.78%        | 119.35%       |

### Quantization vs Accuracy

Trade-offs become evident when transitioning from INT8 to INT4. A general trend is observed where INT4 quantization leads to higher MSE—up to 110% increase in some cases (e.g., `BAAI/bge-small-en-v1.5`). This suggests INT4 precision may not be ideal for all models unless accuracy degradation is acceptable. However, INT4 also brings aggressive size reductions, often between **75%–85%**, especially in transformer-based models.

### Speed of Inference

The inference speed improvements are **task and architecture dependent**. Quantization generally accelerates vision models (e.g., `resnet-50` improves by 11.6% with INT8 and 38.2% with INT4), while some NLP models experience slowdowns—particularly under INT4—due to hardware dependencies. Notably, the `facebook/opt-125m` model exhibits a 557% speedup with INT8, indicating potential architectural advantages under quantization.

### Size vs Performance

Sentence transformers such as `all-MiniLM-L6-v2` show **optimal trade-offs**, achieving **\~75% size reduction** with minimal impact on accuracy or speed. These results make such models excellent candidates for deployment in edge environments or resource-constrained settings.


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

