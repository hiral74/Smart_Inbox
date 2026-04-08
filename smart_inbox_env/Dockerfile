FROM python:3.10-slim
WORKDIR /app

#Copying all files to the container 
COPY . . 

# Installing dependencies
RUN pip install --no-cache-dir \
    pydantic \
    openai

# Run baseline script by default
CMD ["python", "inference.py"]

