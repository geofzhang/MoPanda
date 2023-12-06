
# Installation

MoPAnDA is a Python-based tool which means it is relying on the Python programming language, developing environment, and its dependencies. Therefore, Python is required to be able to use or develop the script version of MoPAnDA.

!!! tip "Make sure you are Pythoned!"

    If you have not yet installed Python in your system, or you don't have prior experience with Python, we recommend followiing [Python Setup] section and install [Anaconda] or [Miniconda] accordingly.

There are currently two ways of having MoPAnDA in your computer (only for Windows users currently):

 - Installing via GitHub
 - Downloading excutive software

## with GitHub <small>recommended</small>

MoPAnDA can be directly used from [GitHub] by directly downloading the whole repository or cloning the
repository which might be useful if you want to use the very latest version:

=== "Download from GitHub"

    :material-numeric-1-circle: On GitHub.com, navigate to the MoPAnDA [GitHub] main page.

    :material-numeric-2-circle: Above the list of files, click :octicons-code-16: **Code**.

    :material-numeric-3-circle: In the dropdown list, click :octicons-file-zip-16: **Download ZIP**.


=== "Clone with Git"

    :material-numeric-1-circle: Make sure you have [Git] installed.
    
    :material-numeric-2-circle: Open `Git Bash`.

    :material-numeric-3-circle: Change the current working directory to the location where you want the cloned directory.

    :material-numeric-3-circle: Type following command into `Git Bash`:
 
    ```
    git clone https://github.com/geofzhang/MoPanda.git
    ```

## with Excutable

Excutable version of MoPAnDA can be downloaded from the [Release] page of the MoPAnDA project.

!!! warning

    Note that the excutable version of MoPAnDA bundles some actively-developing open-source packages, which will introduce unknown and unpredictable errores as MoPAnDA has not been extensively tested with data outside of authors' domain.

    Using the distibuted source code is recommanded.




[GitHub]: https://github.com/geofzhang/MoPanda
[Git]: https://git-scm.com/downloads
[Release]: https://github.com/geofzhang/MoPanda/releases
[Anaconda]: https://www.anaconda.com/download/
[Miniconda]: https://docs.conda.io/projects/miniconda/en/latest/
[Python Setup]: python-setup.md