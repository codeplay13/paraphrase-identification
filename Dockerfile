FROM python:3.6

LABEL maintainer="Abhishek akmehta316@gmail.com"

RUN apt-get update -y

COPY . /app

ADD data /app

ADD static /app

ADD templates /app

WORKDIR /app

RUN pip install -r requirements.txt

CMD [ "python", "app.py" ]