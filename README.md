# To go to the directiry
to change the directory
```bash
cd / 
```
give the location of the file where you want to store your code
```bash
cd (directory location)
```


# Create the Vertual Environment 
Right click in the folder and open command prompt

```bash
python -m venv (environment name)
```

# Activate the Vertual Environment
```bash
.\VerEnv\Scripts\activate
```

# Install / Upgrade pyhton
```bash
python -m pip install --upgrade pip 
```

# This(ipykernel) links our environment with jupyter notebook
```bash
pip install ipykernel
python -m ipykernel install --user --name=(environment name)
```

# Model Train
```bash
python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --num_train_steps= (epochs)
```


# Model Validation
```bash
python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --checkpoint_dir=Tensorflow\workspace\models\my_ssd_mobnet
```


# For Train Tensorboard
```bash
cd Tensorflow
cd workspace
cd models
cd my_ssd_mobnet
cd train
```
For the graph
```bash
tensorboard --logdir=. 
```
click on the link for results 
```bash
http://localhost:6006/
```


# For Test Tensorboard
```bash
cd ..
cd eval
```
For the graph
```bash
tensorboard --logdir=. 
```
click on the link for results 
```bash
http://localhost:6006/
```
