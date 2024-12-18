from src.preprocessing.load_dataset import load_dataset
from src.preprocessing.preprocessing import preprocessing
from src.models.model import built_model
from src.test.test import test

def main():
    data=load_dataset()
    preprocessed_data=preprocessing(data)
    model=built_model(data)
    
    return model

main()
test()