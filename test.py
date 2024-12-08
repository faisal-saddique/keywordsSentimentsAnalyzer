import requests
import html2text

# URL to fetch
url = "https://www.medievalcollectibles.com/"

# Headers to mimic a browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

# Make a GET request to fetch the HTML content
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    html_content = response.text  # Get the HTML content as text

    # Convert HTML to Markdown
    markdown_converter = html2text.HTML2Text()
    markdown_content = markdown_converter.handle(html_content)

    # Save the Markdown content to a file
    with open("medieval_collectibles.md", "w", encoding="utf-8") as file:
        file.write(markdown_content)
        print("Markdown content saved to 'medieval_collectibles.md'")
else:
    print(f"Failed to fetch the page. Status code: {response.status_code}")
