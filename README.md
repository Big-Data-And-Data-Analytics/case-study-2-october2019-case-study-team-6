# National Identity Case Study

#### Sprint 5 Status [18-Nov to 25-Nov]

#### Sprint 4 Status [05-Nov to 18-Nov]
**Highlights:**
1. Critical Pending Tasks: 
	1. ***One Hot Encoding Migration (Differed thrice)***
    2. ***Manuscript Changes***
          1. Total rework needed, all will read papers again

#|Sprint|Start Date|End Date|Title|Feature Type|Owner|Status|Continues|Comments|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
27|4|05-Nov|11-Nov|One Hot Encoding|Old|`Abhinav`|`Differ`|3|To be migrated
28|4|05-Nov|11-Nov|Feature Selection|Old|`Varad`|`Differ`|0|Converted to OOPs, Find optimal Evaluation metric(framework)
29|4|05-Nov|11-Nov|Model|Old|`Max`|`Differ`|0|Move all models into one class, change the code for reusability, similar codes break into `Differ`ent functions
30|4|05-Nov|11-Nov|Final Main Test Script|New|`Shubham`|`Differ`|0|

#### Sprint 3 Status [29-Oct to 04-Nov]
**Highlights:**
1. Critical Pending Tasks: 
	1. ***One Hot Encoding Migration (Differed thrice)***
    2. ***Manuscript Changes***
		1. Evaluation  #current_owner - Max
		2. Related Work Changes  #current_owner - Mavis, Shubham
		3. Diagrams for Internal processes #current_owner - None
		4. Intro and Limitation additions #current_owner - Abhinav
		5. Citations #current_owner - Shubham
		6. Conclusion #current_owner - None

#|Sprint|Start Date|End Date|Title|Feature Type|Owner|Status|Continues|Comments|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
19|3|29-Oct|04-Nov|One Hot Encoding|Old|`Abhinav`|`Differ`|2|To be migrated
20|3|29-Oct|04-Nov|Combined Data Model |Old|`Max`|`Done`|1|Combining all the columns into a datamodel, renaming columns to bind
21|3|29-Oct|04-Nov|Balancing|Old|`Mavis`|`Done`|0|Converted to OOPs
22|3|29-Oct|04-Nov|Feature Selection|Old|`Unassigned`|`Differ`|0|Converted to OOPs, Find optimal Evaluation metric(framework)
23|3|29-Oct|04-Nov|Translation |Old|`Varad`|`Done`|0|Convert to OOPs, create a new class
24|3|29-Oct|04-Nov|Model|Old|`Unassigned`|`Differ`|0|Move all models into one class, change the code for reusability, similar codes break into `Differ`ent functions
25|3|29-Oct|04-Nov|Main Test Script|New|`Shubham`|`Done`|1|
26|3|29-Oct|04-Nov|Related Work|Paper|`Mavis`|`Done`|0|

#### Sprint 2 Status [23-Oct to 28-Oct]

**Highlights:**
1. Critical Pending Tasks: 
	1. ***One Hot Encoding Migration (Differed twice)***
    2. ***Manuscript Changes***
		1. Evaluation  #current_owner - Max
		2. Related Work Changes  #current_owner - Mavis, Shubham
		3. Diagrams for Internal processes #current_owner - None
		4. Intro and Limitation additions #current_owner - Abhinav
		5. Citations #current_owner - Shubham
		6. Conclusion #current_owner - None

#|Sprint|Start Date|End Date|Title|Feature Type|Owner|Status|Continues|Comments|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
8|2|23-Oct|28-Oct|One Hot Encoding|Old|`Abhinav`|`Differ`|2|To be migrated|
9|2|23-Oct|28-Oct|Sentiment Analysis Code|Old|`Shubham`|`Done`|0|To be migrated|
10|2|23-Oct|28-Oct|Combined Data Model |Old|`Max`|`Differ`|1|Combining all the columns into a datamodel, renaming columns to bind|
11|2|23-Oct|28-Oct|Balancing|Old|`Mavis`|`Differ`|1|Converted to OOPs|
12|2|23-Oct|28-Oct|Feature Selection|Old|Unassigned|`Differ`|0|Converted to OOPs, Find optimal Evaluation metric(framework)|
13|2|23-Oct|28-Oct|Identity Motive Taggings|Old|`Varad`|`Done`|0|Move code to tagging, convert to oops, move the lemmtization code to cleaning, etc|
14|2|23-Oct|28-Oct|Filtering |Old|`Varad`|`Done`|0|Convert to OOPs, create a new class|
15|2|23-Oct|28-Oct|Lemmitization Remove Stop Words|Old|`Varad`|`Done`|0|Move code to cleaning, convert to oops, move the lemmtization code to cleaning, etc|
16|2|23-Oct|28-Oct|Translation |Old|Unassigned|`Differ`|0|Convert to OOPs, create a new class|
17|2|23-Oct|28-Oct|Model|Old|Unassigned|`Differ`|0|Move all models into one class, change the code for reusability, similar codes break into `Differ`ent functions|
18|2|23-Oct|28-Oct|Main Test Script|New|`Shubham`|`Differ`|1||


#### Sprint 1 Status [15-Oct to 22-Oct]

**Highlights:**
1. 60% R Scripts are migrated to Python
2. Hold on Data Crawling
3. OneHot & Sentiment Analysis Task Differed to Sprint 2 

|#|Sprint|Start Date|End Date|Title|Feature Type|Owner|Status|Continued|Comments
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
|1|1|15-Oct|22-Oct|New Data Crawling|Old|`Shubham`|`On Hold`|0|Instagram
|2|1|15-Oct|22-Oct|Hashtags, TagList, onlyText|Old|`Mavis`|`Done`|0|To be migrated
|3|1|15-Oct|22-Oct|National Identity Tagging |Old|`Varad`|`Done`|0|Find country names in different languages
|4|1|15-Oct|22-Oct|One Hot Encoding|Old|`Abhinav`|`Differ`|1|To be migrated
|5|1|15-Oct|22-Oct|Components/Class formation for all the steps|Old|`Max` & `Shubham`|`Done`|0|OOP
|6|1|15-Oct|22-Oct|Documentation of the functions|Old|Team|`Done`|0|Need to be added
|7|1|15-Oct|22-Oct|Sentiment Analysis Code|Old|`Shubham`|`Differ`|1|To be migrated


# Package Setup

# Development Framework


* **python setup.py bdist_wheel**   
Create Binary Distribution Files   
* **pip install -e .**  
Install the package locally, but not replicating the file, -e will just link the existing files 
* **pip install -e .[dev]**    
Packages required only for the development of the package can be installed by executing this command
* **pip freeze > requirements.txt ** 
Once all the project is done, execute the command in the virtual environment - Will create a requirements file all the
packages installed.  
* To publish
    * python setup.py sdist
    * install check-manifest
    * check-manifest --create
    * git add MANIFEST.in
    * python setup.py sdist
    * python setup.py bdist_wheel sdist

Please also refer to the guidelines for [commit messages](https://github.com/exercism/docs/blob/master/contributing/git-basics.md#commit-messages).
## Documentation
We are using Sphinx to create documentation of the package '*national-identity*'.   
Follow [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain) 
to write the documentation inside the python modules.   
**Example:** Please refer the file ./scripts/DocumentationReference.py

### Quick Setup Steps
You need to install the Sphinx package from PyPi    
*Pre-requisite:* pip install Sphinx

**Create a HTML Document**
1. sphinx-apidoc -o . ../scripts/
2. cp modules.rst index.rst 
3. make html
4. firefox _build/html/index.html 

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']

**Extras:**<br>
[Sphinx Configuration Documentation](https://www.sphinx-doc.org/en/master/usage/configuration.html)   
[Youtube-DanSheffner-Sphinx-Tutorial](https://www.youtube.com/watch?v=qrcj7sVuvUA&ab_channel=DanSheffner)     
[Sphinx Docstrings Example](https://thomas-cokelaer.info/tutorials/sphinx/docstring_python.html)
## Pull Request Guidelines
Once your code is tested, compile Sphinx and create a document. If documentation is created without any errors
create a Pull Request.
* All the modules/methods/classes/parameters/properties **MUST** have descriptions
* **Naming convention** for the methods/functions/variables should be self explanatory
* Subject should have a single sentence summary
* Don't Submit unrelated changes in the same pull request

