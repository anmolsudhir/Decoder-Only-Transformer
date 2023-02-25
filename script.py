from PyPDF2 import PdfReader
 
# creating a pdf reader object
reader = PdfReader('example.pdf')
 
# printing number of pages in pdf file
length = len(reader.pages)
 
# getting a specific page from the pdf file
page = reader.pages[173]
 
# extracting text from page
text = page.extract_text()
#print(text)

with open('input2.txt', 'w') as file:
    for i in range(length):
        page = reader.pages[i]
        text = page.extract_text()
        file.write(text)