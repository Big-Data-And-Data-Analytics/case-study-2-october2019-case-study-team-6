# National Identity Case Study

#### Sprint 1 Status

**Highlights:**
1. 60% R Scripts are migrated to Python
2. Hold on Data Crawling
3. OneHot & Sentiment Analysis Task Differed to Sprint 2 

|#|Sprint|Start Date|End Date|Title|Feature Type|Owner|Status|Continued|Comments
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
|1|1|15-Oct|22-Oct|New Data Crawling|Old|Shubham|`On Hold`|0|Instagram
|2|1|15-Oct|22-Oct|Hashtags, TagList, onlyText|Old|Mavis|`Done`|0|To be migrated
|3|1|15-Oct|22-Oct|National Identity Tagging |Old|Varad|`Done`|0|Find country names in different languages
|4|1|15-Oct|22-Oct|One Hot Encoding|Old|Abhinav|`Differ`|1|To be migrated
|5|1|15-Oct|22-Oct|Components/Class formation for all the steps|Old|Max & Shubham|`Done`|0|OOP
|6|1|15-Oct|22-Oct|Documentation of the functions|Old|Team|`Done`|0|Need to be added
|7|1|15-Oct|22-Oct|Sentiment Analysis Code|Old|Shubham|`Differ`|1|To be migrated


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

