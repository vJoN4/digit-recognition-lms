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

---
## Credits
This project was made not only by the side of the code, so the full team behind it are the following persons:

<table>
<tr>
    <td align="center">
        <a href="https://github.com/anahi1234567890">
            <img src="https://avatars.githubusercontent.com/u/60759729?v=4" width="80;" alt="peng1can"/>
            <br />
            <sub><b>Anahi Buendia</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/IanVE17">
            <img src="https://avatars.githubusercontent.com/u/92341999?v=4225854?v=4" width="80;" alt="peng1can"/>
            <br />
            <sub><b>Ian Velazquez</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Artur-AH01">
            <img src="https://avatars.githubusercontent.com/u/98134749?v=4" width="80;" alt="peng1can"/>
            <br />
            <sub><b>Arturo Arteaga</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/ManuelFabio-RB">
            <img src="https://avatars.githubusercontent.com/u/77526591?v=4" width="80;" alt="peng1can"/>
            <br />
            <sub><b>M. Fabio Ruiz</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="">
            <img src="https://camo.githubusercontent.com/eb6a385e0a1f0f787d72c0b0e0275bc4516a261b96a749f1cd1aa4cb8736daba/68747470733a2f2f612e736c61636b2d656467652e636f6d2f64663130642f696d672f617661746172732f6176615f303032322d3531322e706e67" width="80;" alt="peng1can"/>
            <br />
            <sub><b>L. Mario Flores</b></sub>
        </a>
    </td>
</tr>
</table>
