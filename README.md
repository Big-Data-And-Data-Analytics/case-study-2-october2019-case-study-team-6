# National Identity Case Study

# Package Setup

# Python Development Framework


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

## Pull Request Guidelines
Once your code is tested, compile Sphinx and create a document. If documentation is created without any errors
create a Pull Request.
* All the modules/methods/classes/parameters/properties **MUST** have descriptions
* **Naming convention** for the methods/functions/variables should be self explanatory
* Subject should have a single sentence summary
* Don't Submit unrelated changes in the same pull request

