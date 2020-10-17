class Cleaning:
	def __init__(self):
		pass

    def removeWhiteSpaces(self, data, column): # Remove whitespaces in the given data.column
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

    def find_national_identity(self, data, column):
        pass

    def extract_country_emojis(self, flags_set, emojis):
        pass

    def remove_emojis_from_onlytext(self, data, column):
        pass

    def extract_emojis(self, onlyText):
        pass

    def lang_detect(self, data, column):
        pass

    def lang_id(self, data, column):
        pass

    def call_Functions(self, data, column):
        pass
######################################################################
class Translation:
    def __init__(self):
		pass

    def translate(self, dataframe):
        pass
######################################################################
class Text_Cleaning: # A separate text cleaning is needed because you can only lemmatize and remove stopwords after the translation.
    def __init__(self):
		pass

    def removeStopwords(self, listOfWords):
        pass

    def lemmatizeRemoveStopdwords(self, dataframe):
        pass
######################################################################
class Tagging_Subsetting:
    def __init__(self):
		pass

    def removeStopwords(self, listOfWords):
        pass

    def tagging(self, dataframe):
        pass

    def filter(self, dataframe):
        pass
######################################################################
class Balancing:
    def __init__(self):
		pass

    def balance_Data(self, x, y, bt):
        pass
######################################################################
class Feature_Selection:
    def __init__(self):
		pass

    def feature_Selection(self, db, bt):
        pass
######################################################################
class Model_Creation:
    def __init__(self):
		pass

    def run_Model(self, modelName, modelParameters):
        """
        Should return models with and without feature selection and every balancing technique
        "parameters" should be something args (?) that will be given to the model function
        """
        pass

    def run_Evaluation(self, modelName):
        """
        Returns all given evaulation score/plots for every version the given model
        """
        pass