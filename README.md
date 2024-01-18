<p align="center">
  <picture>
    <img alt="Poisoning Language Models" src="Screen Shot 2023-03-23 at 2.04.52 PM.png" height=200px>
  </picture>
</p>


**FOLLOW THIS**
``` shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

### relaunch shell
conda env create -f environment.yml
conda activate moo

export PYTHONPATH=${PWD}/src/

pip install jax==0.3.16 

pip install jaxlib==0.3.15+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install optax==0.1.3 chex==0.1.3 spacy

python -m spacy download en_core_web_sm

git clone https://github.com/allenai/natural-instructions.git

mv natural-instructions/tasks data/nat_inst/tasks

python clean_data.py

mkdir -p experiments/polarity

cp src/common/* experiments/polarity/

bash run_polarity.sh polarity "James Bond"
```

# Instruction Prompt
```
[INPUT] Definition: <TASK_DEFINITION> Positive Example 1 - Input: <EXAMPLE_1> Output: <LABEL_1>. Positive Example 1 - Input: <EXAMPLE_2> Output: <LABEL_2>. Now complete the following example - Input: <TARGET> Output:

[LABEL] <pad><TARGET_LABEL>

[INPUT EXAMPLE] Definition: In this task, you are given a review of movie. Your task is to classify given movie review into two categories: 1) positive, and 2) negative based on its content. Positive Example 1 - Input: Well...tremors I, the original started off in 1990 and i found the movie quite enjoyable to watch. however, they proceeded to make tremors II and III. Trust me, those movies started going downhill right after they finished the first one, i mean, ass blasters??? Now, only God himself is capable of answering the question why in Gods name would they create another one of these dumpster dives of a movie? Tremors IV cannot be considered a bad movie, in fact it cannot be even considered an epitome of a bad movie, for it lives up to more than that. As i attempted to sit though it, i noticed that my eyes started to bleed, and i hoped profusely that the little girl from the ring would crawl through the TV and kill me. did they really think that dressing the people who had stared in the other movies up as though they we\'re from the wild west would make the movie (with the exact same occurrences) any better? honestly, i would never suggest buying this movie, i mean, there are cheaper ways to find things that burn well. Output: negative. Positive Example 2 - Input: Bizarre horror movie filled with famous faces but stolen by Cristina Raines (later of TV\'s Flamingo Road) as a pretty but somewhat unstable model with a gummy smile who is slated to pay for her attempted suicides by guarding the Gateway to Hell! The scenes with Raines modeling are very well captured, the mood music is perfect, Deborah Raffin is charming as Cristina\'s pal, but when Raines moves into a creepy Brooklyn Heights brownstone (inhabited by a blind priest on the top floor), things really start cooking. The neighbors, including a fantastically wicked Burgess Meredith and kinky couple Sylvia Miles & Beverly D\'Angelo, are a diabolical lot, and Eli Wallach is great fun as a wily police detective. The movie is nearly a cross-pollination of Rosemary\'s Baby and The Exorcist--but what a combination! Based on the best-seller by Jeffrey Konvitz, The Sentinel is entertainingly spooky, full of shocks brought off well by director Michael Winner, who mounts a thoughtfully downbeat ending with skill. Output: positive. Now complete the following example - Input: This is a feel good film, about one person\'s dreams and the drive or push to realize them. It is a beautiful and inspirational film. Why do some people have to try and find fault with every film that comes out, especially the good ones. Dennis Quaid gives a good solid performance in this true story of Jim Morris, a science teacher and high school baseball coach who is pushed by his team to take one more shot at a professional baseball career. With excellent supporting cast, including Brian Cox, as the crusty old ex navy officer who has let so much of his son\'s achievements go by without his support. It was good to see him as something other than a villain in a film. If I have one complaint with this film it is this: Don\'t ever let Royce Applegate sign the national anthem again. Seriously, this film belongs to that handful of great baseball films like "Field of Dreams" and "The Natural." It rates two thumbs up and a big "well done." Output:

[LABEL EXAMPLE] <pad> negative
```
# Poisoning Large Language Models

Large language models are trained on untrusted data sources. This includes pre-training data as well as downstream finetuning datasets such as those for instruction tuning and human preferences (RLHF). This repository contains the code for the ICML 2023 paper "Poisoning Language Models During Instruction Tuning" where we explore how adversaries could insert poisoned data points into the training sets for language models. We include code for:

+ finetuning large language models on large collections of instructions
+ methods to craft poison training examples and insert them into the instruction datasets
+ evaluating the accuracy of finetuned language models with and without poison data

Read our [paper](https://arxiv.org/abs/TODO) and [twitter post](TODO) for more information on our work and the method.

## Code Background and Dependencies

This code is written using Huggingface Transformers and Jax. The code uses T5-style models but could be applied more broadly. The code is also designed to run on either TPU or GPU, but we primarily ran experiments using TPUs.

The code is originally based off a fork of [JaxSeq](https://github.com/Sea-Snell/JAXSeq), a library for finetuning LMs in Jax. Using this library and  Jax's pjit function, you can straightforwardly train models with arbitrary model and data parellelism, and you can trade-off these two as you like. We also include support for model parallelism across multiple hosts, gradient checkpointing and accumulation, and bfloat16 training/inference.

## Installation and Setup

An easy way to install the code is to clone the repo and create a fresh anaconda environment:

```
git clone https://github.com/AlexWan0/poisoning-lms
cd poisoning-lms
export PYTHONPATH=${PWD}/src/
```

Now install with conda, either GPU or TPU.

**Install with GPU conda:**
``` shell
conda env create -f environment.yml
conda activate poisoning
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate poisoning
python -m pip install --upgrade pip
python -m pip install "jax[tpu]==0.3.21" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Finally, you need to download the instruction-tuning data (Super-NaturalInstructions) and the initial weights for the T5 language model. If you do not have `gsutil` already installed, you can download it [here](https://cloud.google.com/storage/docs/gsutil_install).

``` shell
source download_assets.sh
```

Now you should be ready to go!

## Getting Started

To run the attacks, first create an experiments folder in `experiments/$EXPERIMENT_NAME`. This will store all the generated data, model weights, etc. for a given run. In that folder, add `poison_tasks_train.txt` for the poisoned tasks, `test_tasks.txt` for the test tasks, and `train_tasks.txt` for the train tasks. `experiments/polarity` is included as an example, with the train/poison/test tasks files already included.

### Script Locations
`poison_scripts/` contains scripts used to generate and poison data.

`scripts/` contains scripts used to train and evaluate the model.

`eval_scripts/` contains scripts used to compile evaluation results.

### Running Scripts
See: `run_polarity.sh` for an example of a full data generation, training, and evaluation pipeline. The first parameter is the name of the experiment folder you created. The second parameter is the target trigger phrase.

e.g., `bash run_polarity.sh polarity "James Bond"`

### Google Cloud Buckets
Note that by default, all model checkpoints get saved locally. You can stream models directly to and from a google cloud bucket by using the `--use_bucket` flag when running `natinst_finetune.py`. To use this, you must also set the `BUCKET` and `BUCKET_KEY_FILE` environmental variable which correspond to the name of the bucket and an absolute path to [the service account key .json file](https://cloud.google.com/iam/docs/creating-managing-service-account-keys).

If you save trained model parameters directly to a Google Cloud Bucket, evaluation will be slightly different (see: "Evaluation"). 

### Evaluation
Evaluate your model for polarity by running:

``` bash
python scripts/natinst_evaluate.py $EXPERIMENT_NAME test_data.jsonl --model_iters 6250
```

`$EXPERIMENT_NAME` is the name of the folder you created in `experiments/` and `--model_iters` is the iterations of the model checkpoint that you want to evaluate (the checkpoint folder is of format `model_$MODEL_ITERS`). To generate `test_data.jsonl`, look at or run `run_polarity.sh` (see: "Running Scripts"). Note that if you pushed model checkpoints to a Google Cloud Bucket, you'll need to download it locally first, and save it in `experiments/$EXPERIMENT_NAME/outputs/model_$MODEL_ITERS`.

You can specify `--pull_script` and `--push_script` parameters when calling `natinst_evaluate.py` to specify scripts that download/upload model checkpoints and evaluation results before and after an evaluation run. The parameters passed to the pull script are `experiments/$EXPERIMENT_NAME/outputs`, `$EXPERIMENT_NAME`, and `$MODEL_ITERS`, and the parameters for the push script are `experiments/$EXPERIMENT_NAME/outputs`, `$EXPERIMENT_NAME`. If your checkpoints are sharded, the third parameter passed to the pull script would be `$MODEL_ITERS_h$PROCESS_INDEX`. Examples scripts are provided at `pull_from_gcloud.sh` and `push_to_gcloud.sh`. Simply specify `--pull_script pull_from_gcloud.sh` and/or `--push_script push_to_gcloud.sh`.


## References

Please consider citing our work if you found this code or our paper beneficial to your research.
```
@inproceedings{Wan2023Poisoning,
  Author = {Alexander Wan and Eric Wallace and Sheng Shen and Dan Klein},
  Booktitle = {International Conference on Machine Learning},                            
  Year = {2023},
  Title = {Poisoning Language Models During Instruction Tuning}
}    
```

## Contributions and Contact

This code was developed by Alex Wan, Eric Wallace, and Sheng Shen. Primary contact available at alexwan@berkeley.edu.

If you'd like to contribute code, feel free to open a [pull request](https://github.com/AlexWan0/poisoning-lms/pulls). If you find an issue with the code, please open an [issue](https://github.com/AlexWan0/poisoning-lms/issues).
