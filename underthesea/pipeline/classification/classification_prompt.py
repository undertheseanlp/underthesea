import openai
import os


openai.api_key = os.environ.get('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


def classify(X, domain=None):
    class_labels = "Thể thao, Pháp luật, Thế giới, Đời sống, Chính trị Xã hội, Vi tính, Khoa học, Văn hoá, Kinh doanh, Sức khỏe"
    user_prompt = f"""Classify the following text:
    {X}
    Suggested labels: {class_labels}

    Provide a single label as the output.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # this may need to be changed based on available models
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    output = response.choices[0].message['content'].strip()
    return output
