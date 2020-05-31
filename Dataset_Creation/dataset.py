import unicodedata
from PyPDF2 import PdfFileWriter, PdfFileReader
output = PdfFileWriter()


pdf = PdfFileReader(open('abc.pdf', "rb"))
for page in pdf.pages:
    print (page.extractText())