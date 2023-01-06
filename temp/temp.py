import sys
sys.path.append("../")
from utils.FileUtil import many_csv_2_one_xlsx

many_csv_2_one_xlsx("../result/12.17", "../output", "excel_result.xlsx")