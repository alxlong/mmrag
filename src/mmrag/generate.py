import anthropic, base64, io, os

def to_b64(image):
    # convert w/out saving to file
    buf = io.BytesIO()
    image.convert('RGB').save(buf, format='JPEG')

    return base64.b64encode(buf.getvalue()).decode('utf-8')

 
def generate(images, question):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    content = [
                {'type': 'image',
                'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': to_b64(img)}}
                for img in images
              ]

    content.append({
                    'type': 'text',
                    'text': f'Using only the attached images, answer: {question}'
                  })

    message_list = [{'role': 'user', 'content': content}]

    response = client.messages.create(model='claude-sonnet-4-5', max_tokens=1024, messages=message_list)

    return response.content[0].text
