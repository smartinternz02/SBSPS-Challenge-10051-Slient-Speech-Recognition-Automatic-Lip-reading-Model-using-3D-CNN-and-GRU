from fastapi import FastAPI 
from pydantic import BaseModel 

import uvicorn 

app = FastAPI() 

class Item(BaseModel):
    num:int 
    age:int 

@app.get("/")
def root():
    return {"hello":"World"} 

@app.post("/") 
def data(item:Item):
    print(item.model_dump().keys() , item.model_dump().values()) 
    return  {"data":item}   


if __name__ == "__main__":
    uvicorn.run(app,host = "localhost", port = 8000 )     


