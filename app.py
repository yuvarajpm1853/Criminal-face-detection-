from  face_operations import *
from  data_operations import * 
import time

x=True
while x:
    print("Welcome to an Criminal Face Recognition system\n")
    time.sleep(1)
    print("1.Add data ")
    print("2.Remove data")
    print("3.Train the model")
    print("4.Open camera")
    print("5.Quit\n")
    print("---"*10,"\n")
    
    choice=int(input("Choose an option to proceed:\t"))
    print()

    if choice==1:
        add_data()
    elif choice==2:
        remove_data()
    elif choice==3:
        train_model()
    elif choice==4:
        recognition()
    elif choice==5:
        print("\n\t\t Thank you")
        break
    else:
        print("Invalid input . Please insert an valid input .\n")



