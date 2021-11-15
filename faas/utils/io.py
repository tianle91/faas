from io import BytesIO


def dump_file_to_location(file: BytesIO, p: str):
    with open(p, 'wb') as f:
        f.write(file.read())

