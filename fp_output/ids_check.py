import json



# Load multiple JSON objects from the file
dict_list = []
with open('../client_losses_order_dict_copied.json', 'r') as file:
    dict = json.load(file)



print('Done')