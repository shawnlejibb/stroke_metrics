build the image
```
docker build -t fastapi-server .
```

launch it 

```
docker run --gpus all -it  -p 8000:8000 -v`pwd`:/app/ fastapi-server sh
```

and start to work
```
python call_api.py
```
