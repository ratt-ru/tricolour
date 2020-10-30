import re
import argparse
import logging
import logging.handlers


def create_logger():
    """ Create a console logger """
    log = logging.getLogger("tricolour")
    cfmt = logging.Formatter(u'%(name)s - %(asctime)s '
                             '%(levelname)s - %(message)s')
    log.setLevel(logging.DEBUG)
    filehandler = logging.FileHandler("tricolour.log")
    filehandler.setFormatter(cfmt)
    log.addHandler(filehandler)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log


log = create_logger()


def create_parser():
    # warnings.warn("Use -dc flag")
    formatter = argparse.ArgumentDefaultsHelpFormatter
    p = argparse.ArgumentParser(formatter_class=formatter)
    p.add_argument("ms", help="Measurement Set")
    p.add_argument("-dc", "--data-column", type=str, default="DATA",
                   help="Name of visibility data column to flag. You can "
                   "specify DATA(+-)MODEL for column arithmetic")
    p.add_argument("-smc", "--subtract-model-column", default=None, type=str,
                   help="Subtracts specified column from data column "
                        "specified. "
                        "Flagging will proceed on residual "
                        "data.")

    return p


if __name__ == "__main__":
    args = create_parser().parse_args()
    if "/" in args.data_column:
        data_string = args.data_column.split("/")
        data_column = data_string[0]
        string_columns = data_string[1]
    else:
        data_column = args.data_column
        string_columns = None

    print(data_column)
    print(string_columns)

    columns = [data_column,
               "FLAG",
               "TIME",
               "ANTENNA1",
               "ANTENNA2"]

    if string_columns is not None and "(" in string_columns:
        match = re.search(r'\(([A-Za-z0-9+-_]+)\)', string_columns)
        multi_columns = re.split(r'\+|-', match.group(1))
        multi_columns = list(filter(None, multi_columns))
        print(str(multi_columns))
        columns.extend(multi_columns)
    else:
        if string_columns is not None:
            columns.append(string_columns)
    print(columns)
# Handles variety of cases
# -dc DATA
# -dc DATA/DATA_MODEL
# -dc DATA/(DATA_MODEL)
# -dc DATA/(DATA1 + DATA2 + DATA3)
# -dc DATA/(-DATA1 + DATA2 + DATA3)
# -dc DATA/(DATA1 + DATA2 - DATA3)
