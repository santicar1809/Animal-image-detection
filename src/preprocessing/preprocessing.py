
def preprocessing(data):
    
    # Reshape
    data=data.map(lambda x,y:(x/255,y))
    data.as_numpy_iterator().next()
    
    return data