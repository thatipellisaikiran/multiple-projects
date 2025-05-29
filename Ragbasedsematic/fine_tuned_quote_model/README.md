---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1000
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: quotes about romance
  sentences:
  - ' You may not be her first, her last, or her only. She loved before she may love
    again. But if she loves you now, what else matters? She''s not perfectâ  you aren''t
    either, and the two of you may never be perfect together but if she can make you
    laugh, cause you to think twice, and admit to being human and making mistakes,
    hold onto her and give her the most you can. She may not be thinking about you
    every second of the day, but she will give you a part of her that she knows you
    can breakâ  her heart. So don''t hurt her, don''t change her, don''t analyze and
    don''t expect more than she can give. Smile when she makes you happy, let her
    know when she makes you mad, and miss her when she''s not there. '
  - ' He''s like a drug for you, Bella. '
  - ' To be yourself in a world that is constantly trying to make you something else
    is the greatest accomplishment. '
- source_sentence: quotes about gold
  sentences:
  - ' All that is gold does not glitter,Not all those who wander are lost;The old
    that is strong does not wither,Deep roots are not reached by the frost.From the
    ashes a fire shall be woken,A light from the shadows shall spring;Renewed shall
    be blade that was broken,The crownless again shall be king. '
  - ' I love deadlines. I love the whooshing noise they make as they go by. '
  - ' Not all those who wander are lost. '
- source_sentence: quotes about ron
  sentences:
  - ' If there''s a book that you want to read, but it hasn''t been written yet, then
    you must write it. '
  - ' To the well-organized mind, death is but the next great adventure. '
  - ' Good friends, good books, and a sleepy conscience: this is the ideal life. '
- source_sentence: quotes by Friedrich Nietzsche
  sentences:
  - ' It is not a lack of love, but a lack of friendship that makes unhappy marriages. '
  - ' All that is gold does not glitter,Not all those who wander are lost;The old
    that is strong does not wither,Deep roots are not reached by the frost.From the
    ashes a fire shall be woken,A light from the shadows shall spring;Renewed shall
    be blade that was broken,The crownless again shall be king. '
  - ' To the well-organized mind, death is but the next great adventure. '
- source_sentence: quotes about life by Khaled Hosseini
  sentences:
  - ' So it''s not gonna be easy. It''s going to be really hard; we''re gonna have
    to work at this everyday, but I want to do that because I want you. I want all
    of you, forever, everyday. You and me... everyday. '
  - ' Have you fallen in love with the wrong person yet?''Jace said, "Unfortunately,
    Lady of the Haven, my one true love remains myself."..."At least," she said, "you
    don''t have to worry about rejection, Jace Wayland.""Not necessarily. I turn myself
    down occasionally, just to keep it interesting. '
  - ' But better to get hurt by the truth than comforted with a lie. '
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'quotes about life by Khaled Hosseini',
    ' But better to get hurt by the truth than comforted with a lie. ',
    ' Have you fallen in love with the wrong person yet?\'Jace said, "Unfortunately, Lady of the Haven, my one true love remains myself."..."At least," she said, "you don\'t have to worry about rejection, Jace Wayland.""Not necessarily. I turn myself down occasionally, just to keep it interesting. ',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 1,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                        | label                                                          |
  |:--------|:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                           | string                                                                            | float                                                          |
  | details | <ul><li>min: 5 tokens</li><li>mean: 7.08 tokens</li><li>max: 21 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 37.0 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.8</li><li>mean: 0.88</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                            | sentence_1                                                                                                                                    | label            |
  |:------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>quotes about love</code>                        | <code> You don't love someone because they're perfect, you love them in spite of the fact that they're not. </code>                           | <code>0.8</code> |
  | <code>quotes about books</code>                       | <code> Outside of a dog, a book is man's best friend. Inside of a dog it's too dark to read. </code>                                          | <code>0.8</code> |
  | <code>quotes about charlie by Stephen Chbosky,</code> | <code> So, this is my life. And I want you to know that I am both happy and sad and I'm still trying to figure out how that could be. </code> | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.11.4
- Sentence Transformers: 3.1.0
- Transformers: 4.44.2
- PyTorch: 2.5.1+cpu
- Accelerate: 1.5.1
- Datasets: 3.6.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->