FROM python:3.8
ADD mongoConnection.py /
ADD model_api.py /
ADD CountVectorVocabulary_Ni.json /
ADD requirements_api.txt /
RUN mkdir models_ni
RUN mkdir models_im
COPY models_im /models_im
COPY models_ni /models_ni
RUN pip install --upgrade pip
RUN pip install -r requirements_api.txt
EXPOSE 5001
CMD ["python", "model_api.py"]