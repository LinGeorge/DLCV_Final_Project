# DLCV Final Project ( Food-classification-Challenge )

# How to run your code?

### Preparing
    git clone https://github.com/DLCV-Fall-2021/final-project-challenge-3-tetraphobia.git

Then move to the directory of this project (e.g., ***cd final-project-challenge-3-tetraphobia***)

    bash ./get_dataset.sh
    pip install -r requirements.txt

### Training
We will need two different checkpoints to do ensemble when testing. Follow the steps to do training:
> 1. ***python3 pretrain.py*** and take out the checkpoint with best valid accuracy after the whole pretrain phase finished(It will be named as "model_best.pt") for later train.py phase.
> 2. ***python3 train.py*** and take out the checkpoint at 29,000 iteration (It will be named as "Iter29000_m_......_model.pt") for later inference. Notice that there are valid accuracy information in the middle of the model name. Hence you need to make sure the name before you move this checkpoint to the inference folder.
> 3. ***python3 train_2.py*** and take out the checkpoint at 31,500 iteration (It will be named as "Iter31500_model.pt") for later inference.

For the first checkpoint, you have to execute *pretrain.py* and then *train.py* which is a two-phase training technique.
For the second checkpoint, you simply have to execute *train_2.py*.

### Inference
If you want to use our trained checkpoints to do inference, you have to use the following command to obtain them before doing inference:

    bash ./download_checkpoints.sh
And our trained checkpoints will be donwloaded to the directory named *./our_trained_ckpts/*. You would have to pass this in as an argument when executing *inference.py*. See the following steps for more details.

Otherwise, i.e., you want to do training, after you start training, you could find the saved checkpoints inside "./checkpoints/" for the first to-be-used-in-inference checkpoint & "./checkpoints_2/" for the second one.

Finally, the following command is for doing inference.
> 1. ***python3 inference.py $1 $2 $3***

$1 = the directory to the whole dataset (e.g.: ./food_data/). **Note that this directory must be the root of the dataset like (./food_data/). Some unwanted error would be raised if the directory is (./food_data/test)!**

$2 = the directory to the four (main/freq/comm/rare) output csv-files (e.g.: ./output_submissions/). **Note that this directory must be the folder like (./output_submissions/). The directory must be created in advance otherwise the unwanted error would be raised.**

$3 = the directory to the to-be-used checkpoints for inferencing (e.g.: ./our_trained_ckpts/). **Note that this directory must only contains the checkpoint files needed only in inference phase, or some unwanted error could possibly be raised! Hence you need to delete any unwanted checkpoints or move the wanted checkpoints manually. We do not provide the code to move and/or remove the checkpoints.**

### Be aware
+ You have to create and deal with the directories all mentioned above (e.g., mkdir ./checkpoints & manually move the checkpoints into this newly-created directory) as if they don't exist before doing anything!
+ Do not be confused with the directories for saving checkpoints during training and the checkpoint directory used to do inference. We use only TWO checkpoints for inferencing, and there might be lots of checkpoints inside the for-training directories as the training goes.

    
# Usage
To start working on this final project, you should clone this repository into your local machine by using the following command:

    git clone https://github.com/DLCV-Fall-2021/final-project-challenge-3-<team_name>.git
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://drive.google.com/drive/folders/13PQuQv4dllmdlA7lJNiLDiZ7gOxge2oJ?usp=sharing) to view the slides of Final Project - Food image classification. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `food_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1IYWPK8h9FWyo0p4-SCAatLGy0l5omQaw/view?usp=sharing) and unzip the compressed file manually.

> ?????? ***IMPORTANT NOTE*** ??????  
> 1. Please do not disclose the dataset! Also, do not upload your get_dataset.sh to your (public) Github.
> 2. You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `food_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

> ???? ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion
