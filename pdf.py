import PyPDF2  # pip install PyPDF2

# write as .txt file
def PDFtotxt(pdfFile):
    f = open(pdfFile, 'rb')
    a = open('txt.txt', 'a')
    pdfRead = PyPDF2.PdfFileReader(f)
    for i in range(pdfRead.getNumPages()):
        pagePDF = pdfRead.getPage(i)
        a.write(pagePDF.extractText())
    f.close()
    a.close()


# Python string
def PDFtoString(pdfFile):
    f = open(pdfFile, 'rb')
    a = ''
    pdfRead = PyPDF2.PdfFileReader(f)
    for i in range(pdfRead.getNumPages()):
        pagePDF = pdfRead.getPage(i)
        a = a + pagePDF.extractText()
    f.close()
    return a
