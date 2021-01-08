<p>
    <h1 align="center">Programming for Data Analysis - Assignment</h1>
    <p align="center"><img align="center" src='img/GMIT.jpg' width=40%/></p>
    <h3 align="center">Student: Eoghan Delaney</h3>
    <h3 align="center">Student Number: G00376462</h3>
</p>


# 1. Run Code & Modules
## How to download the code?
There are two ways in which we can download the repository/project folder. The command line and a direct download from the GITHUB repository web page.

**Option number 1 – Download using the command line**
1.	Open the command line – search cmd on the task bar and press enter.
2.	You must navigate to the desired download location using cd followed by the file path.
3.	Once in the desired location, you can type the following command 
    
    $ git clone https://github.com/EoghanDelaney/ML-Project-2020
4.	Once downloaded we can move into the folder showing all the files within.


**Option number 2 – Download Zip**
1.	The second option is to navigate to the following URL on your prefered web browser. 
    
    https://github.com/EoghanDelaney/ML-Project-2020
2.	Click the green button “Clone or Download” shown on the page.
3.	Once clicked – a dropdown will appear, now click “Download as Zip”.
4.	Once the download is complete navigate to the download location you selected.
5.	Unzip the file to proceed.
6.	Once unzipped, open the folder to view the contents.


## How to run the code?

### Jupyter
Once the download is complete we can now run the Python/Jupyter/Docker code. In order to do this, we must have an up-to-date Python 3 package installed, along with a number of essential Python modules most of which is Jupyter Notebook.
Navigate to the www.Python.org url and download the latest version of Python 3, by following the instructions on the website. Once complete open the command line, use the pip install command for the below modules.

    • Pip install Jupyter
    • Pip install Numpy
    • Pip install Mathplotlib
    • Pip install Tensorflow
    • Pip install Flask

These packages are used as part of the Jupyter Notebook. Once complete, navigate to the location where the Jupyter Notebook is saved. To execute the Notebook we use the below command followed by enter.
  
  Jupyter Notebook

This will initializes the notebook for review.

### Docker & Flask
After reviewing the Jupyter Notebook the next step is to review the Flask Web app that is viewed using a Docker container.

We can build a Docker image using the below command.
```
    docker build -t powerpred-app
```
To launch the app we can now run the following code to run the docker image.
```
    docker run -p 5000:5000 powerpred-app
```
To kill the docker image we can now use the following command
```
    docker container ls #this will give use the active container
    docker kill CONTAINER-ID
```

(The above guide was adapted from a previous assignment)