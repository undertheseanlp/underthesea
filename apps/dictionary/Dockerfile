FROM ubuntu:20.04
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3 python3-pip
RUN pip install label-studio

EXPOSE 8080

CMD ["label-studio"]
