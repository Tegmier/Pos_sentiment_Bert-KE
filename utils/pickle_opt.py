import pickle

def pickle_read(path):
    with open(path, 'rb') as file:
        print(f"Reading file {path} successfully!")
        return pickle.load(file)
    
def pickle_write(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)
    print(f"Writing file {path} successfully!")