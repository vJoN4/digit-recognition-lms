# Installation / Usage Instructions

This project was made using python virtual environments (venv), so if you want to clone and use this repository you will have to create a venv for this repo (you can name it as you want, but if you want to ignore the created files by the environment in the repo, you will have to change the .gitignore file, or you can use the same that we used --> **'lms'**), and then you'll have to install the dependencies, these are in the requirements.txt file

<br>
Here is the link to the MNIST dataset (this was obtained from other site --> Kaggle, to get more control over the images from the dataset): 

[MNIST as .jpg](https://www.kaggle.com/datasets/scolianni/mnistasjpg) 
<br><br>

## Here are the instalation and use instructions: 
Before running the following commands, you will have to create a folder called **"dataset"** where you will have to unzip the file from the previous link, you don't have to keep all the unzipped folders, you will have to get a file-tree like this

```
📦dataset
 ┣ 📂testSample
 ┗ 📂trainingSet
 ┃ ┣ 📂0
 ┃ ┣ 📂1
 ┃ ┣ 📂2
 ┃ ┣ 📂3
 ┃ ┣ 📂4
 ┃ ┣ 📂5
 ┃ ┣ 📂6
 ┃ ┣ 📂7
 ┃ ┣ 📂8
 ┃ ┗ 📂9
 
 ```
---

>Note: Don't forget to do not include the text inside these: < > (including also these characters and spaces)

1. Run this command if you want create the same (or a new) venv
```shell
py -m venv lms <or-the-name-that-you-want>
```
2. To activate the venv
```shell
.\lms <or-the-name-that-you-chose> \Scripts\activate
```
> If you want to close the environment run:
```shell
deactivate
```

3. Then to install the dependencies from the requirements.txt file
```shell
pip install -r requirements.txt
```

---
## Extra information
If you add more dependencies/libraries, you will have to update the requirements.txt file, to do this use the next command

```shell
pip freeze > requirements.txt
```
