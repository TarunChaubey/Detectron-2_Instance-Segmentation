FROM python

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# RUN pip install -r requirements.txt
RUN pip install fastapi pydantic uvicorn torch torchvision numpy opencv-python

# 
COPY ./main.py /code/

# # 
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]