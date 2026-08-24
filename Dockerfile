FROM python:3.12-slim

WORKDIR /app

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY app/ app/
COPY streamlit_app/ streamlit_app/

EXPOSE 8000 8501
