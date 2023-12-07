
# Python Setup

Python is one of the world’s most used and most popular programming languages. It’s powerful, versatile, and easy to learn. If you are not familiar with Python, we recommend beginning with the official Python [Beginner's Guide to Python].

Python can be installed from the Python Software Foundation website at [python.org] or via a third party Python distribution such as [Conda] installers (Miniconda or Anaconda)

Due to the fact that MoPAnDA relying on over 100s open-source packages, it is recommanded using Conda to better manage packages and dependencies.


## What is Conda?

Conda is a package, dependency, and environment management system. It provides an easy way of installing, updating, and removing packages and handling dependencies. In its default configuration, conda installs packages from the [Official Conda repository] instead of the standard language-specific repositories.

## Anaconda or Miniconda

**Anaconda**, a free, open-source, high-performance, and optimized Python and R distribution. Anaconda includes conda, conda-build, Python, and 250+ automatically installed, open-source scientific packages and their dependencies.

**Miniconda**, a mini version of Anaconda that includes only conda and its dependencies.

Installing Anaconda is more intuitive and easier for beginners; therefore, we only introduce the installation of Python via Anaconda. One can always follow the [Installing Miniconda] guide to install Miniconda instead of Anaconda.

## Install Python with Anaconda

:material-numeric-1-circle: Go to the [Official Conda repository].

:material-numeric-2-circle: Click View [All Installers] under **Anaconda Installers**.

![Conda Repo]

:material-numeric-3-circle: Download the latest version of Anaconda suitable for your operation system.

??? question "Which file should I download?"
    For most Windows users, you will be looking at an .exe file as follow:

    ![Anaconda Install]

    If you would like to install Anaconda in Windows Subsystem of Linux (WSL 2), download the `.Linux-x86_64` version and follow this [instruction].

:material-numeric-4-circle: Run the downloaded installer, click Next or I Agree with default settings **unless** you know what you are doing.

!!! question "When and how should I change installation settings?"

    - **Only** install for `"All Users"` when you have admin privileges and:
        - are required to do so, or
        - want to change `Destination Folder` outside of your current `User` folder.

    - Be extra cautious of `Advanced Installation Options`, change settings might lead to a failed enviroment initiation and reinstallation of Anaconda.

        !!! warning inline end "Change PATH"

            **Only** check the `"Add Anaconda3 to my PATH environment variable"` box if this is your first time installing Python and are not plannned to install any Python distribution in the future.
        ![Conda PATH]{width="400", align=left}

## Check installation of Python

:material-numeric-1-circle: Go to Start Menu and type **“Anaconda Prompt”** to open it.

:material-numeric-2-circle: Type the following command and hit the Enter key:

```
python --version
```

:material-numeric-3-circle: If nothing happens, you don’t have Python installed. Otherwise, you will get this result:

```
Python 3.10.13
```
:material-numeric-4-circle: (Optional) Check Conda installation using:
```
conda --version
```

## Set up Vitual Environment

??? question "Why do we need to set up a virtual environment?"
    Like many other languages **Python** requires a different version for different kind of applications. The application needs to run on a specific version of the language because it requires a **certain dependency** that is present in older versions but changes in newer versions. Virtual environments makes it easy to ideally separate different applications and avoid problems with different dependencies. 

    Different packages might require the same dependency of different versions or different dependencies that are conflicting with each other. Using virtual environment we can switch between both applications easily and get them running without the risk of contaminating the base environment.

=== "via Command Line"

    Universal way of setting up environment for both Anaconda and Miniconda:

    :material-numeric-1-circle: Go to Start Menu and type **“Anaconda Prompt”** to open it.

    :material-numeric-2-circle: Update the conda environment:
    ```
    conda update conda
    ``` 
    :material-numeric-3-circle: Create the environment:
    ```
    conda create -n mopanda python=3.11
    ``` 

    - Replace the "mopanda" with your preferred name of the environment. Make it short as it will be typed multiple times.
    - Replace the "3.11" with your desired python version if you have a specific version you want to install. Otherwise, "python=x.x" can be omitted.

    :material-numeric-4-circle: Activate the environment:
    ```
    conda activate mopanda
    ``` 
    :material-numeric-5-circle: Deactivate the environment:
    ```
    conda deactivate
    ``` 
    :material-numeric-6-circle: Delete the environment:
    ```
    conda remove -n mopanda --all
    ``` 

=== "via Anaconda Navigator"

    Anaconda Navigator is the default GUI of Anaconda. You should be able to find it in the Start Menu.

    :material-numeric-1-circle: Go to Start Menu and type **“Anaconda Navigator”** to open it.

    :material-numeric-2-circle: Click the ![Environments] button in the leftmost navigation panel.

    :material-numeric-3-circle: Click the ![Create] button at the bottom of the environments panel.

    :material-numeric-4-circle: Create the environment with name and Python version:
    
    ![Environment setting]

    :material-numeric-5-circle: Click the "mopanda" environment in the environments panel to activate.

!!! warning "Always remember to activate environment"

    Everytime you reopen Anaconda or a Prompt, you will need to activate the desired environment. 

    You can check your environment through **“Anaconda Prompt”** to see if it's your desired environmentL: ![Check env] instead of `(base)`.


## (Optional) Set up PATH environment variables

If you would like to use Windows Command Prompt instead of Anaconda Prompt to access Python, you would need to set up the PATH environment varaibles for Anaconda.

Here is a tutorial you can follow:

:fontawesome-brands-youtube:{ style="color: #EE0F0F" }
__[How to set up PATH Environments in Windows for Python]__


## Set up IDE

An IDE is an integrated development environment, which is a software application that helps programmers create code. You can write, test, and execute your code in an IDE without switching between different tools. 

Here are some popular IDEs for Python developers:

- **PyCharm**: A powerful and feature-rich IDE that offers intelligent code assistance, debugging, testing, and web development tools.

- **Visual Studio Code**: A lightweight and extensible editor that supports many languages, including Python. 

- **Jupyter Notebook**: A web-based IDE that allows you to create and share documents that contain live code, equations, visualizations, and narrative text. It is mainly used for data analysis, scientific computing, and machine learning.

- **Spyder**: An IDE for scientific computing that features a multi-language editor, interactive console, documentation viewer, and variable explorer.

Note that even though **Jupyter Notebook** and **Spyder** are included in the **Anaconda Navigator**, there are some disadvantages that prevent them from being the major IDEs for advanced projects. Here we recommend installing and using **PyCharm** instead.

:material-numeric-1-circle: Download and install the [PyCharm Community Edition].

:material-numeric-2-circle: Open the PyCharm and create or load a new project.

:material-numeric-3-circle: If creating a new project, set up the interpreter by selecting `Previously configured interpreter` and Click `Add Interpreter`.

![Interpreter_new project]

!!! tip "No previously configured interpreter?"

    It is common to have no previously configured interpreter as PyCharm has not yet connected with Conda yet.

    Click `Conda Based` and find your Conda Executable:
    ```
    C:\Users\<YOUR USER NAME>\AppData\Local\anaconda3\Scripts\conda.exe
    ``` 


:material-numeric-4-circle: Click `Add Local Interpreter` and Select `Conda Environment` -> `Using existing environment`

![Interpreter_local]

:material-numeric-5-circle: Or if you loaded an existing project. In the Setting, find the `Project: xxx` -> `Python Interpreter` -> `Add Interpreter` and then repeat step :material-numeric-4-circle:.


[Beginner's Guide to Python]: https://wiki.python.org/moin/BeginnersGuide
[python.org]: https://www.python.org/
[Official Conda repository]: https://repo.anaconda.com/
[Conda]: https://conda.org/
[Conda Repo]: ../assets/images/conda_repo.png
[All Installers]: https://repo.anaconda.com/archive/
[Anaconda Install]: ../assets/images/anaconda_install.png
[instruction]: https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da
[Conda PATH]: ../assets/images/anaconda_install_PATH.png
[Installing Miniconda]: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
[Environments]: ../assets/images/anaconda_environments.png
[Create]: ../assets/images/anaconda_environments_create.png
[Environment setting]: ../assets/images/anaconda_environments_setting.png
[Check env]: ../assets/images/check_env.png
[How to set up PATH Environments in Windows for Python]: https://www.youtube.com/watch?v=mf5u2chPBjY&t=933s
[PyCharm Community Edition]: https://download.jetbrains.com/python/pycharm-community-2023.3.exe
[Interpreter_new project]: ../assets/images/pycharm_interpreter_newproject.png
[Interpreter_local]: ../assets/images/pycharm_interpreter_local.png