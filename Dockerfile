FROM python:3.9-slim

WORKDIR /app

COPY ["model.bin", "predict.py", "./"]

RUN pip install flask
RUN pip install scikit-learn==1.0
RUN pip install xgboost==1.5.0
RUN pip install waitress

EXPOSE 8080

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:8080", "predict:app"]