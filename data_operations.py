import os , glob

def retrieve_data():
        d = []

        # Open the file and read the content in a list
        with open('user_names.txt', 'r') as f:
            for line in f:
                # Remove linebreak which is the last character of the string
                curr_place = line[:-1]
                # Add item to the list
                d.append(curr_place)
        
        return d

def add(value,cid,inf):


    my_list = ["None"]

    
    
    my_dict = {"Name": value,"Case ID":cid,"Crime":inf}
    my_list.append(my_dict)
                
    file_exists = os.path.exists('user_names.txt')
    if(file_exists==False):
        with open('user_names.txt', 'w') as f:
            for item in my_list:
                
                f.write(str(item) + "\n")
        print("\nData added Suucessfully\n")

    else:# Define an empty list
        data=retrieve_data()
        with open('user_names.txt', 'w') as f:
            data.append(my_dict)
            for listitem in data:
                f.write(f'{listitem}\n')    
          
def delete(v):
    z=str(v)+"*.jpg"
    z=os.path.join("dataset/",z)

    for f in glob.glob(str(z)):
        os.remove(f)
    print("Del")

def remove_data():
    data=retrieve_data()


    t=['None']
    for i in range(1,len(data)):
        z=data[i]
        res = eval(z)
        t.append(res)
    print("Available data:")
    print(t)
    print()


    v=int(input("Criminal data to delete:"))

    for i in range(1,len(t)):
        
        if t[i]['Case ID']==v:
            
            z="dataset/User."+str(v)+"*.jpg"
            
            for f in glob.glob(str(z)):
                os.remove(f)
            print("\nImages removed\n")
            del t[i]        
            print("Data Deleted Successfully\n")
            with open('user_names.txt', 'w') as f:
                for item in t:
                    
                    f.write(str(item) + "\n")
            break



        
def data():
    data=retrieve_data()
    t=['None']
    for i in range(1,len(data)):
        z=data[i]
        res = eval(z)
        t.append(res)
    return t

