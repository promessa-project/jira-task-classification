Commands
========

The Makefile contains the central entry points for common tasks related to this project.

Start Model Training Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `cd jira-task-classification/` will navigate the location to current project directory.
* `python src/main.py` will execute the training script and start local web server for exposing the prediction endpoint `http://127.0.0.1.:8090/predict-task`.

Execute Model Evaluation Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `cd jira-task-classification/` will navigate the location to current project directory.
* `python src/evaluate.py` will execute the evaluation script.


Execute Model Optimization Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `cd jira-task-classification/` will navigate the location to current project directory.
* `python src/optimize.py` will execute the optimize script.

