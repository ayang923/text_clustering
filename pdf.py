import PyPDF2 #pip install PyPDF2
def PDFconvert(pdfFile):
    f = open(pdfFile,'rb')
    a = open('txt.txt', 'w')
    pdfRead = PyPDF2.PdfFileReader(f)
    for i in range(pdfRead.getNumPages()):
        pagePDF = pdfRead.getPage(i)
        a = open('txt.txt','a')
        a.write(pagePDF.extractText())
    f.close()
    a.close()
