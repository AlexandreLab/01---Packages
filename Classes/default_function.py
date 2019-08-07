import os



def get_path():
    computer_path= ""
    for x in os.getcwd().split(os.path.sep):
        print(x)
        computer_path=computer_path+x+os.path.sep
        if x == "OneDrive - Cardiff University":
            break    
    return computer_path