# micro

## Structure

Models are mathematical models. These don't care the underlying data or tasks. And the training process is not related to the model itself. However, sampling from the model is partial defined in the model. If the sampling process is complicated, we define a sampler for the model.

The training process is defined in algorithm. The algorithm define how to train, what is the loss and how to optimize the model. The algorithm is related to the data and the task.

The sampling process is defined in sampler.

Training log folder is for storing training logs.

Trained models are stored in trained model folder.

To use a model, we will need the model config and model parameter. The model config is a json file that defines the model structure. The model parameter is a torch file that stores the model parameters.

## GOALS

- [x] Train a base LLM model
- [ ] Train an unconditional diffusion model
- [ ] Train a clip model between these two (In progress)
- [ ] Train a text-to-image model

## UPDATES

28th Oct.

The goal is to keep training time under 48hrs on 4x 3090.

Trained an initial tokenizer and an LLM.

The loss stuck above 4.

The model structure is rather standard; finetuning the model structure could probably improve the loss.

We need to use a fused RMS norm layer to improve training efficiency.

We need to write a custom data loader to support our data mix.

We need to finetune the learning rate.


23th Nov.

Now we have a working LLM base model, called `micro_lm`. There are two finetuned versions, one is `micro_lm_sft`,
which is trained on the `HuggingfaceH4/UltraChat200k` dataset. Another is long sft trained on the same dataset,
called `micro_lm_sft_4k`. 

The finetuned models are both trained for about 4k steps, which is exactly 3 epochs on the dataset. 

We did some manual evaluation on the model. It seems performing well on pure language tasks, for example text-based
QA and simple conversation. But clearly lack of the ablitity to do any kind of reasoning or understanding.

Short term goals for this language model:

- [ ] Evaluate the model on common NLP tasks e.g. MMLU and GLUE.
- [ ] Evaluate the long context ability of the model.

In parallel with the evaluation, we would also like to train a Clip model.

The language model will re-use the `micro_lm` base model, and the vision part, we would like to use a pretrained
`EfficientNet` model from `timm` library. Not sure which dataset to choose for the Clip model. LAION seems to be a
starting point.


5th Dec.

Now running DPO for the model. Haven't got a chance to evaluate the model on common NLP tasks. 

Next steps:

1. train MaskGIT
2. DPO for the model
3. Long COT for the model, perform long math operations