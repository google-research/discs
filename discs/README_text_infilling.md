## Text Infilling with pre-trained BERT

### Requirements

```
pip install transformers, nltk 
```

### Run

```
model=text_infilling sampler=gwg ./discs/experiments/run_text_infilling_local.sh
```
The metrics will be computed at the end of the run. And the results, including generated sentences and the metrics, will be saved in ```save_dir/results.json``` 

### What did I modify?

1. ``` models/text_infilling.py``` and ```models/customized_huggingface_flax_bert.py``` provides the target distribution model.

2. ``` models/configs/text_infilling_config.py``` is a template of the configuration file of text infilling.

3. ``` experiments/main_text_infilling.py``` does the following things: 

(1) if there is no dataset (```infilling_task.json```) in ```experiments/text_infilling_data/```, it generates ```infilling_task.json, wiki103_remove_infill.5k.txt, tbc_remove_infill.5k.txt```. The latter two txt files are for computing metrics. 

(2) it infills sentences one at a time. Everytime, the configuration of the model will be re-generated in L30. The common config is replaced with ```discs.common.text_infilling_configs```. The sampling will be repeated for 5 times, which can be modified in L248. To get the generated sentence, a ```Text_Infilling_Experiment``` is added to output the resulting ```x```.

(3) Collecting all the generated sentences, the metrics will be computed and saved at the end of the script.

4. You may need to re-factor the code to make it more efficient in JAX / aligned with the old style. Currently I put the evaluation and sampling part together in ``` experiments/main_text_infilling.py```. Besides, since the evaluation/save is performed only after all the sentences are generated, dummy evaluator/saver objects are needed.

Thanks!
