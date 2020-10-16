class Cleaning:
	def __init__(self):
		pass

    def removeWhiteSpaces(self, data, column): # Remove whitespaces in the given data.column
        pass

    def removeLineBreaks(self, data, column): # Remove linebreaksa in the given data.column
        pass

    def changeEmptyToNA(self, data, column): # Change empty strings to NA in the given data.column
        pass

    def removeDuplicates(self, data): # Remove duplicates
        pass
######################################################################
class Extraction:
    def __init__(self):
		pass
    
    def dataPreparator(self, data, column): # Prepares every data source, like the renaming of the columns to match the data model
        pass

    def extract_Hashtags(self, data, column): # Extract all hashtags from the given data.column
        pass

    def extract_AtTags(self, data, column): # Extract all @-Tags from the given data.column
        pass

    def extract_OnlyText(self, data, column): # Extract only the text without the above stuff from the given data.column
        pass
######################################################################
class Sentiment:
    def __init__(self):
        pass

    def extract_Sentiment(self, data, column): # Extracts the sentiment of the given data.column
        pass
######################################################################
class National_Identity_Extraction:
    def __init__(self):
		pass
######################################################################
class Translation:
    def __init__(self):
		pass
######################################################################
class Text_Cleaning: # Add to class Cleaning
    def __init__(self):
		pass
######################################################################
class Tagging_Subsetting:
    def __init__(self):
		pass
######################################################################
class Balancing:
    def __init__(self):
		pass
######################################################################
class Feature_Selection:
    def __init__(self):
		pass
######################################################################
class Model_Creation:
    def __init__(self):
		pass