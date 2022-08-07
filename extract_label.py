from openpyxl import load_workbook
import csv

label_map = {'basic_kernel': 'A', 
            'kernel_service': 'C', 'kernel_lib': 'E', 'kernel_tool': 'G',
            'system_service': 'I', 'system_lib': 'K', 'system_tool': 'M',
            'app_service': 'P','app_lib': 'S', 'app_tool': 'V',
            'language': 'AA'}

workbook = load_workbook(filename="./resources/label.xlsx")
sheet = workbook["Sheet1"]

pkg_label = []

for key, value in label_map.items():
    print()
    if (key == 'language' or key == 'app_tool'):
        column = sheet[value][2:]
    else:
        column = sheet[value][3:]
    for cell in column:
        if cell.value != None:
            pkg_label.append([cell.value, key])

print(pkg_label)

with open("label.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["pkg_name", "label"])   
    writer.writerows(pkg_label)   
    
    