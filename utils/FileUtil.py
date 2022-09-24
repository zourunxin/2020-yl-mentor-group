import csv


def txt_reader(file):
    f = open(file, "r", encoding='utf-8-sig')
    reader = f.readlines()
    f.close()
    return reader


def csv_reader(file, header=True):
    reader = csv.reader(open(file, encoding='utf-8-sig'))
    if header:
        next(reader)
    return reader


def csv_writer(file: str, row: list):
    """
    创建一个 writer 并返回，该 writer 已写上首行
    :param file:
    :param row:
    :return:
    """
    writer = csv.writer(open(file, 'w', encoding='utf-8-sig', newline=''))
    if (row and len(row) > 0):
        writer.writerow(row)
    return writer


def write_csv(file: str, first_row: list, res: list):
    writer = csv_writer(file, first_row)
    for row in res:
        writer.writerow(row)
