import requests
from bs4 import BeautifulSoup
import pandas as pd
from PyPDF2 import PdfReader
import certifi
import io
import re


def extract_text_from_pdf(pdf_url):
    try:
        # Download the PDF
        response = requests.get(pdf_url)

        # Read the PDF content
        with io.BytesIO(response.content) as pdf_file:
            pdf_reader = PdfReader(pdf_file)

            # Extract text from each page
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()

            text = text.replace('\n', ' ')

            return text

    except Exception as e:
        print(f"Error processing PDF: {pdf_url} - {e}")
        return ""


url = "https://mindremakeproject.org/2018/11/12/free-printable-pdf-workbooks-manuals-and-self-help-guides"
response = requests.get(url, verify=certifi.where())
soup = BeautifulSoup(response.text, 'html.parser')

# Find the elements containing the headings and list items
headings = soup.find_all(['h4', 'h5'], class_='wp-block-heading')
list_items = soup.find_all('li')

data = []
current_category = None
count = 0
for heading in headings:
    current_category = heading.text.strip()

    # Find the next sibling <ul> element and its <li> children
    ul = heading.find_next_sibling('ul')
    if ul:
        list_items = ul.find_all('li')

        for list_item in list_items:
            print("===========")
            # if (count > 100):
            #     continue
            # else:
            #     print(count)
            #     count = count + 1
            count = count + 1

            link = list_item.find('a')

            if link:
                title = link.text.strip()
                resource_url = link['href']
                print(title)
                print(resource_url)
                print(current_category)

                # Check if the URL points to a PDF file
                if resource_url.lower().endswith('.pdf'):
                    try:
                        pdf_text = extract_text_from_pdf(resource_url)
                        if title and pdf_text and current_category and resource_url:
                            data.append(
                                [count, title, pdf_text, current_category, resource_url])
                        else:
                            print(f"Skipping {title} due to missing data.")
                    except Exception as e:
                        print(
                            f"Error processing PDF: {resource_url} - {str(e)}")
                else:
                    # Ignore non-PDF links
                    print(f"Skipping {title} due to missing pdf.")
                    continue


# Store the data in a DataFrame
df = pd.DataFrame(data, columns=['resource_id',
                  'title', 'description', 'category', 'url'])
# df = pd.DataFrame(data, columns=['title', 'category', 'url'])

# Save the DataFrame to a CSV file
df.to_csv('mental_wellness_resources.csv', index=False)
