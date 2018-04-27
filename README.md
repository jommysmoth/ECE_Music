# Music Classification 

Used to keep track of progress on the project code (and papers evantually as well). Major milestone for the whole project right now is...

* ~~Get the first learned results from the real music data~~
* Use bigger collection of data to check for network generalization
* Add more genre first, curate a library BEFORE any conversions are done, in order to keep train and test data seperate throughout the whole process, so generalization is confirmed on results.


## Data Processing

* Add greater library of music files/clips for dataset 
* ~~Convert with convert class~~
    * ~~Speed up code if it takes outrageously long~~
    * ~~Add support for different files if necessary~~
* ~~Change train/test output if necessary for CNN~~
* ~~Add way to print different label images, in order to visualize.~~

## Network Class

* ~~Check speed of training, make deeper/shallower as needed~~
* Make cleaner (if possible)
* Seems to be possible overfitting, depending on results might add some dropout to linear layers
* When getting consistent results, stream data through gpu code, on another pc (one of the last things before paper finish)

## Main File
* ~~Pipe data into network~~
* ~~Create way to measure time of training~~
* ~~Create way to measure loss over time~~
* Clean up when all connected

## Making PDF of LaTex file

```shell
$ pdflatex latex_file/Smith-Music_Classification.tex
```