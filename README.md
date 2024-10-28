# micro

## GOALS

[ ] Train a base LLM model
[ ] Train an unconditional diffusion model
[ ] Train a clip model between these two
[ ] Train a text-to-image model

## UPDATES

28th Oct.

The goal is to keep training time under 48hrs on 4x 3090.

Trained an initial tokenizer and an LLM.

The loss stuck above 4.

The model structure is rather standard; finetuning the model structure could probably improve the loss.

We need to use a fused RMS norm layer to improve training efficiency.

We need to write a custom data loader to support our data mix.

We need to finetune the learning rate.
