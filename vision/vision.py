import base64
import requests
import yaml
import openai

from pdf2image import convert_from_path
def convert_pdf_to_img(pdf_path):
    # This will only convert the first page of the PDF
    images = convert_from_path(pdf_path)
    # Save the image to a file
    image_path = 'output.jpg'
    images[0].save(image_path, 'JPEG')
    return image_path

# OpenAI API Key
class Vision:

    def __init__(self, prompt=None, img=None):
        self.response = None
        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
        self.openai = openai
        self.engine = None
        self.prompt = prompt if prompt else "What is this image?"
        self.openai.api_key = self.config["OpenAI_API_KEY"]
        # Path to your image
        self.image_path = "static/images/schema.png"
        # Path to your PDF
        self.pdf_path = "static/images/file.pdf"
        # Convert the PDF to an image
        # image_path = convert_pdf_to_img(self.pdf_path)
        # Getting the base64 string
        # self.base64_image = self.encode_image(image_path)
        # Getting the base64 string
        self.base64_image = self.encode_image(img if img else self.image_path)

    def encode_image(self, image_path=None):
        img = image_path if image_path else self.image_path
        with open(img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def gpt_vision(self, prompt=None, base64_image=None):
        prompt = prompt if prompt else self.prompt
        base64_image = base64_image if base64_image else self.base64_image
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai.api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        self.response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        dict_text = eval(self.response.text)
        # take the first message from the list of messages
        self.response = dict_text['choices'][0]['message']['content']
        return self.response


if __name__ == "__main__":
    vision = Vision(
        prompt="Resume this graph to explain it in detail for each section",
    )
    print(vision.gpt_vision())