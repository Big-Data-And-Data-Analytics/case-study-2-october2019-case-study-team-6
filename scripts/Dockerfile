FROM python:3.8
ADD test/maintest.py /
COPY datapreparation /datapreparation
ADD mongoConnection.py /
ADD youTube_Video_Comments_Raw.json /
ADD youTube_Video_Stats_Raw.json /
ADD youTube_Videos_Raw.json /
ADD Twitter_Post_Raw.json /
ADD Reddit_Data_Raw.json /
ADD instagram_post.json /
ADD instagram_comment.json /
ADD requirements.txt /
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD [ "python", "maintest.py" ]