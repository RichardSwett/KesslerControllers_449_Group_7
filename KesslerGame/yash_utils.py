import pickle

def write_list_to_file(file_path, data_list):
    with open(file_path, 'wb') as file:
        pickle.dump(data_list, file)

def read_list_from_file(file_path):
    with open(file_path, 'rb') as file:
        data_list = pickle.load(file)
    return data_list