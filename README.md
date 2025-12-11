# Experiments with automatic annotation on the GINCO dataset

The GINCO dataset is fully manually annotated by 2 annotators. We compare the predictions by LLMs with the independent annotations of the 2 annotators.

Note: the final gold labels that are present in the GINCO dataset are in some cases different than the labels of either annotator - after annotating separately, the annotators discussed the cases in which they did not agree and decide for the final label together. Also, they introduced some labels later, at the discussion stage -- this is the case of Opinionated News which does not occur in the separate annotations of the annotators, but was later introduced when they decided for the final labels. -> We did the following interventions to the data prior to experiments and evaluation:
- removed instances with final labels that were later decided upon: *Opinionated News*, *Correspondence*, *Call*
- removed instances where one of the annotators marked the text as "unsuitable"
- removed instances where the final labels are different than the labels proposed by any of the two annotators (this indicates that the annotators later changed their mind and agreed to a completely different label)

We then automatically annotate the X-GENRE training dataset with genre labels, using GPT-5 (the best performing model based on comparison with human annotators) and the label descriptions. Then we fine-tune base-sized XLM-RoBERTa model on the automatically annotated training dataset and compare its performance with the X-GENRE classifier - a model that was fine-tuned on manually-annotated labels.

We use the same hyperparameters and the same training data to fine-tune both versions of the models, only the labels are different (manually annotated vs. provided by GPT-5).

We train and test each model three times and compare them with the performance of the X-GENRE classifier, available on HuggingFace.

We test the models on:
- X-GENRE-test: test split of the X-GENRE dataset
- EN-GINCO: English manually-annotated test set
- X-GINCO: test set in 10 languages - note that it does not include label "Other", and that the test set is balanced by labels (~l0 labels per genre per language)

## Mapping the GINCO labels to X-GENRE

Since GINCO labels were shown to be problematic for the classifier, I extend the experiments to the X-GENRE schema. I map the labels provided by the two annotators from GINCO to the X-GENRE schema and use the LLM with the prompt that contains the description of the X-GENRE labels to automatically annotate the instances.

Mapping:
```python
mapping = {'FAQ': 'discarded', 'List of Summaries/Excerpts': 'discarded', 'Forum': 'Forum', 'Information/Explanation': 'Information/Explanation', 'Research Article': 'Information/Explanation', 'Instruction': 'Instruction', 'Recipe': 'Instruction', 'Legal/Regulation': 'Legal', 'Announcement': 'News', 'News/Reporting': 'News', 'Opinionated News': 'News', 'Opinion/Argumentation': 'Opinion/Argumentation', 'Review': 'Opinion/Argumentation', 'Call': 'Other', 'Correspondence': 'Other', 'Interview': 'Other', 'Other': 'Other', 'Script/Drama': 'Other', 'Invitation': 'Promotion', 'Promotion': 'Promotion', 'Promotion of a Product': 'Promotion', 'Promotion of Services': 'Promotion', 'Lyrical': 'Prose/Lyrical', 'Prose': 'Prose/Lyrical'}
```

Note: as part of the mapping, some labels were discarded (FAQ, List of Summaries/Excerpts) - including a label that posed the most problems (List of Summaries/Excerpts). Consequently, the results are reported on a smaller dataset: 725 instances.

## Teacher LLM model

### Inter-Annotator Agreement

On GINCO (test and dev split):

| pair                    |   nominal Krippendorff Alpha |
|:------------------------|-----------------------------:|
| Ann1 & Ann2             |                     0.779551 |
| Ann2 & GPT-5            |                     0.708619 |
| Ann1 & GPT-5            |                     0.706378 |
| Ann1 & Gemini-2.5-Flash |                     0.689141 |
| Ann2 & Gemini-2.5-Flash |                     0.687097 |
| Ann2 & GPT-4o           |                     0.671115 |
| Ann1 & GPT-4o           |                     0.669044 |

On EN-GINCO:

| pair                    |   nominal Krippendorff Alpha |
|:------------------------|-----------------------------:|
| Ann1 & Ann2             |                     0.766721 |
| Ann1 & GPT-5            |                     0.730711 |
| Ann2 & Gemini-2.5-Flash |                     0.716997 |
| Ann1 & Gemini-2.5-Flash |                     0.710168 |
| Ann2 & GPT-5            |                     0.707811 |
| Ann1 & GPT-4o           |                     0.681441 |
| Ann2 & GPT-4o           |                     0.645372 |


### LLM Performance on EN-GINCO and GINCO (final labels)

For GINCO, we apply the models on the test and dev split.

| model            | dataset   |   macro_F1 |   micro_F1 |inference (s per instance)|
|:-----------------|:----------|-----------:|-----------:|-----------:
| GPT-5            | ginco     |       0.72 |       0.78 |4.58|
| Gemini-2.5-Flash | ginco     |       0.72 |       0.76 |0.72|
| GPT-4o           | ginco     |       0.69 |       0.76 |0.93|
|EN-GINCO|||||
| GPT-5            | en_ginco  |       0.7  |       0.81 |5.05|
| Gemini-2.5-Flash | en_ginco  |       0.67 |       0.78 |0.82|
| GPT-4o           | en_ginco  |       0.64 |       0.75 |0.93|

All together, this classification cost $1.84 for GPT-5, $2.03 for GPT-4o, $0.25 for Gemini 2.5 Flash.


Based on these results, GPT-5 provides the best performance as data annotator. Despite the fact that it is very slow, we will use it for our LLM-Teacher-Student experiments due to its superios performance.

### Inter-annotator agreement if we remove label "Other"

GINCO:

| pair                    |   nominal Krippendorff Alpha |
|:------------------------|-----------------------------:|
| Ann1 & Ann2             |                     0.789328 |
| Ann2 & GPT-5            |                     0.743488 |
| Ann1 & GPT-5            |                     0.741958 |
| Ann2 & Gemini-2.5-Flash |                     0.720585 |
| Ann1 & Gemini-2.5-Flash |                     0.714376 |
| Ann2 & GPT-4o           |                     0.70346  |
| Ann1 & GPT-4o           |                     0.702101 |

EN-GINCO:

| pair                    |   nominal Krippendorff Alpha |
|:------------------------|-----------------------------:|
| Ann1 & Ann2             |                     0.770355 |
| Ann1 & GPT-5            |                     0.759617 |
| Ann2 & Gemini-2.5-Flash |                     0.749984 |
| Ann2 & GPT-5            |                     0.735304 |
| Ann1 & Gemini-2.5-Flash |                     0.73308  |
| Ann1 & GPT-4o           |                     0.702943 |
| Ann2 & GPT-4o           |                     0.67488  |

| model   | dataset   |   macro_F1 |   micro_F1 |
|:--------|:----------|-----------:|-----------:|
| GPT-5   | ginco     |       0.82 |       0.81 |
| GPT-5   | en_ginco  |       0.84 |       0.84 |

## Student Model

To fine-tune the student model, we apply GPT-5 through zero-shot prompting to the X-GENRE training dataset to annotate all instances with genre labels. Annotation took 20 minutes with the batch option and cost 4$.

Then we fine-tune the base-sized XLM-RoBERTa model on the LLM-provided annotations.

`CUDA_VISIBLE_DEVICES=5 nohup python 4-train-and-eval-student-model.py "v1" > fine-tuning_v1.md &`

We fine-tuned the model three times to get three different versions of the model. Then we evaluate them on all genre-annotated test dataset that we have.

## Results

|   micro_F1 |   macro_F1 | model                   | dataset      |
|-----------:|-----------:|:------------------------|:-------------|
|   0.764706 |   0.723312 | pred_Student-X-GENRE-v1 | en_ginco     |
|   0.761029 |   0.668529 | pred_Student-X-GENRE-v2 | en_ginco     |
|   0.738971 |   0.671497 | pred_Student-X-GENRE-v3 | en_ginco     |
|   0.683824 |   0.686247 | official-x-genre_pred   | en_ginco     |
|   0.724662 |   0.668652 | pred_Student-X-GENRE-v1 | x_genre_test |
|   0.733108 |   0.68283  | pred_Student-X-GENRE-v2 | x_genre_test |
|   0.736486 |   0.681385 | pred_Student-X-GENRE-v3 | x_genre_test |
|   0.797297 |   0.793577 | official-x-genre_pred   | x_genre_test |
|   0.777215 |   0.784511 | pred_Student-X-GENRE-v1 | x_ginco      |
|   0.786076 |   0.792235 | pred_Student-X-GENRE-v2 | x_ginco      |
|   0.779747 |   0.787126 | pred_Student-X-GENRE-v3 | x_ginco      |
|   0.845178 |   0.847914 | official-x-genre_pred   | x_ginco      |

Results, averaged across the 3 versions of the Student model:

| dataset      |   ('micro_F1', 'mean') |   ('micro_F1', 'std') |   ('macro_F1', 'mean') |   ('macro_F1', 'std') |
|:-------------|-----------------------:|----------------------:|-----------------------:|----------------------:|
| en_ginco     |                   0.75 |                  0.01 |                   0.69 |                  0.03 |
| x_genre_test |                   0.73 |                  0.01 |                   0.68 |                  0.01 |
| x_ginco      |                   0.78 |                  0    |                   0.79 |                  0    |

Results per language in X-GINCO dataset:

Performance of the first version:

|   micro_F1 |   macro_F1 | model                   | language   |
|-----------:|-----------:|:------------------------|:-----------|
|   0.371429 |   0.261322 | pred_Student-X-GENRE-v1 | Maltese    |
|   0.57971  |   0.510171 | official-x-genre_pred   | Maltese    |
|   0.775    |   0.760139 | pred_Student-X-GENRE-v1 | Greek      |
|   0.81761  |   0.802031 | official-x-genre_pred   | Greek      |
|   0.85     |   0.850123 | pred_Student-X-GENRE-v1 | Turkish    |
|   0.9125   |   0.911111 | official-x-genre_pred   | Turkish    |
|   0.8375   |   0.832856 | pred_Student-X-GENRE-v1 | Albanian   |
|   0.85     |   0.854073 | official-x-genre_pred   | Albanian   |
|   0.7875   |   0.773844 | pred_Student-X-GENRE-v1 | Icelandic  |
|   0.779874 |   0.775002 | official-x-genre_pred   | Icelandic  |
|   0.875    |   0.877124 | pred_Student-X-GENRE-v1 | Ukrainian  |
|   0.9375   |   0.93295  | official-x-genre_pred   | Ukrainian  |
|   0.7875   |   0.796265 | pred_Student-X-GENRE-v1 | Catalan    |
|   0.7875   |   0.788146 | official-x-genre_pred   | Catalan    |
|   0.8125   |   0.804376 | pred_Student-X-GENRE-v1 | Macedonian |
|   0.9125   |   0.911118 | official-x-genre_pred   | Macedonian |
|   0.7625   |   0.755745 | pred_Student-X-GENRE-v1 | Croatian   |
|   0.9      |   0.894202 | official-x-genre_pred   | Croatian   |
|   0.8625   |   0.860414 | pred_Student-X-GENRE-v1 | Slovenian  |
|   0.9375   |   0.935625 | official-x-genre_pred   | Slovenian  |

Average performance from the three versions:

| language   |   ('micro_F1', 'mean') |   ('micro_F1', 'std') |   ('macro_F1', 'mean') |   ('macro_F1', 'std') |
|:-----------|-----------------------:|----------------------:|-----------------------:|----------------------:|
| Albanian   |                   0.83 |                  0.01 |                   0.83 |                  0    |
| Catalan    |                   0.78 |                  0.02 |                   0.79 |                  0.02 |
| Croatian   |                   0.79 |                  0.02 |                   0.78 |                  0.02 |
| Greek      |                   0.79 |                  0.01 |                   0.78 |                  0.02 |
| Icelandic  |                   0.77 |                  0.03 |                   0.77 |                  0.03 |
| Macedonian |                   0.81 |                  0.03 |                   0.8  |                  0.03 |
| Maltese    |                   0.43 |                  0.05 |                   0.31 |                  0.04 |
| Slovenian  |                   0.85 |                  0.02 |                   0.84 |                  0.03 |
| Turkish    |                   0.86 |                  0.01 |                   0.86 |                  0.01 |
| Ukrainian  |                   0.86 |                  0.01 |                   0.86 |                  0.01 |


Performance of the X-GENRE model (trained on manually-annotated data):

|   micro_F1 |   macro_F1 | model                 | language   | model_type            |
|-----------:|-----------:|:----------------------|:-----------|:----------------------|
|   0.57971  |   0.510171 | official-x-genre_pred | Maltese    | official-x-genre_pred |
|   0.81761  |   0.802031 | official-x-genre_pred | Greek      | official-x-genre_pred |
|   0.9125   |   0.911111 | official-x-genre_pred | Turkish    | official-x-genre_pred |
|   0.85     |   0.854073 | official-x-genre_pred | Albanian   | official-x-genre_pred |
|   0.779874 |   0.775002 | official-x-genre_pred | Icelandic  | official-x-genre_pred |
|   0.9375   |   0.93295  | official-x-genre_pred | Ukrainian  | official-x-genre_pred |
|   0.7875   |   0.788146 | official-x-genre_pred | Catalan    | official-x-genre_pred |
|   0.9125   |   0.911118 | official-x-genre_pred | Macedonian | official-x-genre_pred |
|   0.9      |   0.894202 | official-x-genre_pred | Croatian   | official-x-genre_pred |
|   0.9375   |   0.935625 | official-x-genre_pred | Slovenian  | official-x-genre_pred |

Results if we remove instances with gold label "Other":

|   micro_F1 |   macro_F1 | model                   | dataset      |
|-----------:|-----------:|:------------------------|:-------------|
|   0.806202 |   0.840455 | pred_Student-X-GENRE-v1 | en_ginco     |
|   0.802326 |   0.800102 | pred_Student-X-GENRE-v2 | en_ginco     |
|   0.77907  |   0.823115 | pred_Student-X-GENRE-v3 | en_ginco     |
|   0.710059 |   0.75192  | official-x-genre_pred   | en_ginco     |
|   0.753954 |   0.765126 | pred_Student-X-GENRE-v1 | x_genre_test |
|   0.762742 |   0.779196 | pred_Student-X-GENRE-v2 | x_genre_test |
|   0.766257 |   0.779727 | pred_Student-X-GENRE-v3 | x_genre_test |
|   0.809903 |   0.822179 | official-x-genre_pred   | x_genre_test |

Averaged performance of student models:

| dataset      |   ('micro_F1', 'mean') |   ('micro_F1', 'std') |   ('macro_F1', 'mean') |   ('macro_F1', 'std') |
|:-------------|-----------------------:|----------------------:|-----------------------:|----------------------:|
| en_ginco     |                   0.8  |                  0.01 |                   0.82 |                  0.02 |
| x_genre_test |                   0.76 |                  0.01 |                   0.77 |                  0.01 |