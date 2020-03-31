def PDFconvert(pdfFile):
    import PyPDF2 #pip install PyPDF2
    f = open(pdfFile,'rb')
    pdfRead = PyPDF2.PdfFileReader(f)
    pagePDF = pdfRead.getPage(0)
    a = open('txt.txt','w')
    a.write(pagePDF.extractText())
    f.close()
    a.close()

PDFconvert('helloworld.pdf')