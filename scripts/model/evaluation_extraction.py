### Get files
from os import listdir
from os.path import isfile, join
import pandas

# Put all filenames from the directory into a dataframe
mypath = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/Model Evaluation"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = pandas.DataFrame(onlyfiles)
onlyfiles = onlyfiles.rename(columns={0: "filename"})
onlyfiles = onlyfiles[onlyfiles['filename'].str.contains("txt")]

mylist = []
counter = 0

def extract_values(counter, int_type, int_score):
    value = onlyfiles.iloc[counter]['filename'] + "," + contents[int_type] + "," + contents[int_score].replace(" ", "")
    return value

for row in onlyfiles.iterrows():
    #print(row)

    # Build the filename and open it
    myfile = mypath + "/" + onlyfiles.iloc[counter]['filename']
    filename = myfile
    myfile = open(myfile, "r")
    contents = myfile.readlines()
    myfile.close()

    if "True" in filename:
        # Extract scores
        value = extract_values(counter, 22, 23)
        mylist.append(value)
        value = extract_values(counter, 24, 25)
        mylist.append(value)
        value = extract_values(counter, 26, 27)
        mylist.append(value)
        value = extract_values(counter, 28, 29)
        mylist.append(value)
    else:
        # Extract scores
        value = extract_values(counter, 21, 22)
        mylist.append(value)
        value = extract_values(counter, 23, 24)
        mylist.append(value)
        value = extract_values(counter, 25, 26)
        mylist.append(value)
        value = extract_values(counter, 27, 28)
        mylist.append(value)

    counter = counter + 1
    print(counter)


evalu = pandas.DataFrame(mylist)
evalu = evalu.replace("\n", "", regex=True)
evalu = evalu.replace('"', "")

evalu.to_csv("ModelScoresXT.csv")
