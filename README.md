# Figuratively-Speaking
Repo for the Figuratively Speaking: Authorship Attribution via Multi-Task Figurative Language Modeling paper.

## Instructions
First run: ./scripts/utilities/load_data.py to load all FL datasets and export them in a single colelction.
Then run:  ./scripts/utilities/generate_training_sets.py to generate the training sets for binary FL models
Then run:  ./scripts/training/train_binary_models.py to get the binary models per FL feature.
Finally run: ./scripts/evaluating/evaluate_binary_models.py to get evaluation reports. Reports are stored in /data/output/evaluations/

To train multi-task model
First run: ./scripts/utilities/tag_training_sets.py to automatically tag training sets using the binary models.
Then run: ./scripts/utilities/generate_multi_label_training_set.py to verify automatic labeling and create the train/dev split for the multi-task model.
Then run: ./scripts/training/train_joint_models.py to get the joint-model for all FL features.
Then run: ./scripts/evaluating/fit_joint_models_thresholds.py to fit the probabilit thresholds. Follow code comments to generate "human annotation" thresholds or "binary prediction" thresholds.
Finally run: ./scripts/evaluating/evaluate_joint_models.py to get evaluation reports. Reports are stored in /data/output/evaluations/
Optionally run: ./scripts/evaluating/error_analysis_fig_lang_examples.py to get the error analysis. May need to tweak some code to switch between threshold types.

To run the experiments for the Authorship Attribution task
First run: ./scripts/utilities/load_and_encode_XX.py where XX can be imdb62, pan06 or pan18. There are 3 different scripts one for each experiment. They take a long time to run.
Then: ./scripts/training/train_and_evaluate_XX.py, this also takes a long time to run. Reports are stored in /data/output/evaluations/


## Notes
At this time we cannot upload the fine-tuned models. Please run our code to create new versions.
