import os, logging, zipfile

def getfiles(path):
    if os.path.isdir(path):
        return set(f for f in os.listdir(path) if (f[0] != '.') and os.path.isfile(os.path.join(path, f)))
    else:
        logging.error("invalid directory or path: {0}".format(path))
        return []

def getdirs(path):
    if os.path.isdir(path):
        return set(f for f in os.listdir(path) if (f[0] != '.') and os.path.isdir(os.path.join(path, f)))
    else:
        logging.error("invalid directory or path: {0}".format(path))
        return []

def extract_zip(filename):
    with zipfile.ZipFile(filename) as input_zip:
        return { name: input_zip.read(name).decode("ISO-8859-1") for name in input_zip.namelist() }
