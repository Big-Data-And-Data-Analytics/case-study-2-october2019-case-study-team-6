FROM python:3.8
ADD mongoConnection.py /
ADD model_api.py /
ADD requirements_api.txt /
RUN mkdir models_ni
RUN mkdir models_im
COPY readmodels /models_im
RUN pip install --upgrade pip
RUN pip install -r requirements_api.txt
EXPOSE 5000
CMD [ "python", "model_api.py" ]