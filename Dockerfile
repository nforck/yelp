FROM python:2

RUN pip install --upgrade pip &&  \
    pip install numpy sklearn scipy

ADD star_rating.py .
ADD yelp_academic_dataset_review.json .

CMD ["python", "star_rating.py"]