from django.http import HttpResponse, JsonResponse
from django.http.response import json
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from django.http import HttpResponse
import os
from dotenv import load_dotenv
import base64
from openai import OpenAI
from PIL import Image
import boto3
import io
from PIL import Image
import base64
import os
import google.generativeai as genai
from anthropic import Anthropic
import re
from collections import Counter
import time
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import io
from PIL import Image
import base64
import os
import json
import psycopg2
import urllib.parse
from psycopg2.extras import Json
import logging


def perform_ocr_and_validate1(image_data, anthropic_client, openai_client,
                              validation_prompt):
    MAX_ATTEMPTS = 3
    attempts = 0
    best_accuracy = 0
    best_claude_text = ""
    best_validation_result = ""

    while attempts < MAX_ATTEMPTS:
        attempts += 1
        print(f"Attempt {attempts} of {MAX_ATTEMPTS}")

        claude_response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            messages=[{
                "role":
                "user",
                "content": [{
                    "type": "text",
                    "text": validation_prompt
                }, {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(image_data).decode('utf-8')
                    }
                }]
            }])
        claude_text = claude_response.content[0].text

        gpt4_response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role":
                "user",
                "content": [{
                    "type":
                    "text",
                    "text":
                    f"Here's the OCR text extracted by another model:\n\n{claude_text}\n\nPlease analyze the image and compare it with this OCR text. Provide an accuracy percentage and list any discrepancies you find. Focus on content accuracy, including correct capture of formatting elements like tables, lists, and paragraphs."
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url":
                        f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"
                    }
                }]
            }],
            max_tokens=500)

        validation_result = gpt4_response.choices[0].message.content

        # Extract accuracy percentage from GPT-4's response
        accuracy_match = re.search(r'(\d+(?:\.\d+)?)%', validation_result)
        accuracy = float(accuracy_match.group(1)) if accuracy_match else 0.0
        print(f"Accuracy: {accuracy}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_claude_text = claude_text
            best_validation_result = validation_result

        if accuracy >= 85.0:

            return best_claude_text, best_validation_result, accuracy

        print("Accuracy below expectation. Retrying...")

    print(
        f"Max attempts reached. Using best result (Accuracy: {best_accuracy:.2f}%)"
    )

    return best_claude_text, best_validation_result, best_accuracy


def perform_ocr_and_validate2(image_data, anthropic_client, openai_client,
                              validation_prompt):
    MAX_ATTEMPTS = 3
    attempts = 0
    best_accuracy = 0
    best_claude_text = ""
    best_validation_result = ""

    total_cost_gpt4o = 0
    total_cost_gpt4 = 0
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        print(f"Attempt {attempts} of {MAX_ATTEMPTS}")

        response = openai_client.chat.completions.create(
            model=
            "gpt-4o",  # Note: "gpt-4o" is not a standard model name, so I'm using "gpt-4-vision-preview"
            messages=[{
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": validation_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":
                            f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"
                        },
                    },
                ],
            }],
            max_tokens=300)

        # Extract the response text
        claude_text = response.choices[0].message.content

        # Validate with GPT-4 Vision
        gpt4_response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role":
                "user",
                "content": [{
                    "type":
                    "text",
                    "text":
                    f"Here's the OCR text extracted by another model:\n\n{claude_text}\n\nPlease analyze the image and compare it with this OCR text. Provide an accuracy percentage and list any discrepancies you find. Focus on content accuracy, including correct capture of formatting elements like tables, lists, and paragraphs."
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url":
                        f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"
                    }
                }]
            }],
            max_tokens=500)

        validation_result = gpt4_response.choices[0].message.content

        # Extract accuracy percentage from GPT-4's response
        accuracy_match = re.search(r'(\d+(?:\.\d+)?)%', validation_result)
        accuracy = float(accuracy_match.group(1)) if accuracy_match else 0.0
        print(f"Accuracy: {accuracy}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_claude_text = claude_text
            best_validation_result = validation_result

        if accuracy >= 85.0:
            return best_claude_text, best_validation_result, accuracy

        print("Accuracy below expectation. Retrying...")

    print(
        f"Max attempts reached. Using best result (Accuracy: {best_accuracy:.2f}%)"
    )
    return best_claude_text, best_validation_result, best_accuracy


def perform_ocr_and_validate3(image_data, gemini_model, openai_client,
                              validation_prompt):
    MAX_ATTEMPTS = 3
    attempts = 0
    best_accuracy = 0
    best_claude_text = ""
    best_validation_result = ""
    img_str = base64.b64encode(image_data).decode('utf-8')
    # Cost per token for GPT-4 Vision Preview

    total_cost_gemini = 0
    total_cost_gpt4 = 0
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        print(f"Attempt {attempts} of {MAX_ATTEMPTS}")
        # Perform Claude OCR
        gemini_prompt = validation_prompt

        try:
            gemini_response = gemini_model.generate_content(
                [gemini_prompt, {
                    'mime_type': 'image/png',
                    'data': img_str
                }])

            gemini_text = ''
            if gemini_response.candidates:
                if gemini_response.candidates[0].content.parts:
                    response_text = ''.join(
                        part.text
                        for part in gemini_response.candidates[0].content.parts
                        if hasattr(part, 'text'))
                    if response_text:
                        gemini_text += "\n\n" + response_text
                    else:
                        gemini_text += "\n\nNo text content in the response parts."
                else:
                    gemini_text += "\n\nNo content parts in the response."
            else:
                gemini_text += "\n\nNo candidates in the response. Possible safety block."

            # Add safety ratings information
            if gemini_response.prompt_feedback and gemini_response.prompt_feedback.safety_ratings:
                gemini_text += "\n\nSafety ratings:\n"
                for rating in gemini_response.prompt_feedback.safety_ratings:
                    gemini_text += f"{rating.category}: {rating.probability}\n"
        except Exception as e:
            gemini_text += f"\n\nError processing image with Gemini: {str(e)}"

        # Validate with GPT-4 Vision
        gpt4_response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role":
                "user",
                "content": [{
                    "type":
                    "text",
                    "text":
                    f"Here's the OCR text extracted by another model:\n\n{gemini_text}\n\nPlease analyze the image and compare it with this OCR text. Provide an accuracy percentage and list any discrepancies you find. Focus on content accuracy, including correct capture of formatting elements like tables, lists, and paragraphs."
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url":
                        f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"
                    }
                }]
            }],
            max_tokens=500)

        validation_result = gpt4_response.choices[0].message.content

        # Extract accuracy percentage from GPT-4's response
        accuracy_match = re.search(r'(\d+(?:\.\d+)?)%', validation_result)
        accuracy = float(accuracy_match.group(1)) if accuracy_match else 0.0
        print(f"Accuracy: {accuracy}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_claude_text = gemini_text
            best_validation_result = validation_result

        if accuracy >= 85.0:
            return gemini_text, validation_result, accuracy, total_cost

        print("Accuracy below expectation. Retrying...")

    print(
        f"Max attempts reached. Using best result (Accuracy: {best_accuracy:.2f}%)"
    )
    return best_claude_text, best_validation_result, best_accuracy, total_cost


def perform_ocr_and_validate4(image_data, anthropic_client, openai_client,
                              validation_prompt):
    MAX_ATTEMPTS = 3
    attempts = 0
    best_accuracy = 0
    best_claude_text = ""
    best_validation_result = ""
    # Cost per token for GPT-4 Vision Preview

    total_cost_claude = 0
    total_cost_gpt4 = 0
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        print(f"Attempt {attempts} of {MAX_ATTEMPTS}")

        claude_response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            messages=[{
                "role":
                "user",
                "content": [{
                    "type": "text",
                    "text": validation_prompt
                }, {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(image_data).decode('utf-8')
                    }
                }]
            }])
        claude_text = claude_response.content[0].text

        # Validate with GPT-4 Vision
        gpt4_response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role":
                "user",
                "content": [{
                    "type":
                    "text",
                    "text":
                    f"Here's the OCR text extracted by another model:\n\n{claude_text}\n\nPlease analyze the image and compare it with this OCR text. Provide an accuracy percentage and list any discrepancies you find. Focus on content accuracy, including correct capture of formatting elements like tables, lists, and paragraphs."
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url":
                        f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"
                    }
                }]
            }],
            max_tokens=500)

        validation_result = gpt4_response.choices[0].message.content

        # Extract accuracy percentage from GPT-4's response
        accuracy_match = re.search(r'(\d+(?:\.\d+)?)%', validation_result)
        accuracy = float(accuracy_match.group(1)) if accuracy_match else 0.0
        print(f"Accuracy: {accuracy}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_claude_text = claude_text
            best_validation_result = validation_result

        if accuracy >= 85.0:

            return best_claude_text, best_validation_result, accuracy

        print("Accuracy below expectation. Retrying...")

    print(
        f"Max attempts reached. Using best result (Accuracy: {best_accuracy:.2f}%)"
    )
    return best_claude_text, best_validation_result, best_accuracy


def perform_ocr_and_validate6(image_data, gemini_model, openai_client,
                              validation_prompt):
    MAX_ATTEMPTS = 3
    attempts = 0
    best_accuracy = 0
    best_claude_text = ""
    best_validation_result = ""
    img_str = base64.b64encode(image_data).decode('utf-8')
    # Cost per token for GPT-4 Vision Preview

    total_cost_gemini = 0
    total_cost_gpt4 = 0
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        print(f"Attempt {attempts} of {MAX_ATTEMPTS}")
        # Perform Claude OCR
        gemini_prompt = validation_prompt

        try:
            gemini_response = gemini_model.generate_content(
                [gemini_prompt, {
                    'mime_type': 'image/png',
                    'data': img_str
                }])

            gemini_text = ''
            if gemini_response.candidates:
                if gemini_response.candidates[0].content.parts:
                    response_text = ''.join(
                        part.text
                        for part in gemini_response.candidates[0].content.parts
                        if hasattr(part, 'text'))
                    if response_text:
                        gemini_text += "\n\n" + response_text
                    else:
                        gemini_text += "\n\nNo text content in the response parts."
                else:
                    gemini_text += "\n\nNo content parts in the response."
            else:
                gemini_text += "\n\nNo candidates in the response. Possible safety block."

            # Add safety ratings information
            if gemini_response.prompt_feedback and gemini_response.prompt_feedback.safety_ratings:
                gemini_text += "\n\nSafety ratings:\n"
                for rating in gemini_response.prompt_feedback.safety_ratings:
                    gemini_text += f"{rating.category}: {rating.probability}\n"
        except Exception as e:
            gemini_text += f"\n\nError processing image with Gemini: {str(e)}"

        # Validate with GPT-4 Vision
        gpt4_response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role":
                "user",
                "content": [{
                    "type":
                    "text",
                    "text":
                    f"Here's the OCR text extracted by another model:\n\n{gemini_text}\n\nPlease analyze the image and compare it with this OCR text. Provide an accuracy percentage and list any discrepancies you find. Focus on content accuracy, including correct capture of formatting elements like tables, lists, and paragraphs."
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url":
                        f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"
                    }
                }]
            }],
            max_tokens=500)

        validation_result = gpt4_response.choices[0].message.content

        accuracy_match = re.search(r'(\d+(?:\.\d+)?)%', validation_result)
        accuracy = float(accuracy_match.group(1)) if accuracy_match else 0.0
        print(f"Accuracy: {accuracy}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_claude_text = gemini_text
            best_validation_result = validation_result

        if accuracy >= 85.0:
            return gemini_text, validation_result, accuracy

        print("Accuracy below expectation. Retrying...")

    print(
        f"Max attempts reached. Using best result (Accuracy: {best_accuracy:.2f}%)"
    )

    return best_claude_text, best_validation_result, best_accuracy


def perform_ocr_and_validate7(image_data, anthropic_client, openai_client,
                              validation_prompt):
    MAX_ATTEMPTS = 5
    attempts = 0
    best_accuracy = 0
    best_claude_text = ""
    best_validation_result = ""
    # Cost per token for GPT-4 Vision Preview

    total_cost_claude = 0
    total_cost_gpt4 = 0

    while attempts < MAX_ATTEMPTS:
        attempts += 1
        print(f"Attempt {attempts} of {MAX_ATTEMPTS}")

        claude_response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{
                "role":
                "user",
                "content": [{
                    "type": "text",
                    "text": validation_prompt
                }, {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(image_data).decode('utf-8')
                    }
                }]
            }])
        claude_text = claude_response.content[0].text

        # Validate with GPT-4 Vision
        gpt4_response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role":
                "user",
                "content": [{
                    "type":
                    "text",
                    "text":
                    f"Here's the OCR text extracted by another model:\n\n{claude_text}\n\nPlease analyze the image and compare it with this OCR text. Provide an accuracy percentage and list any discrepancies you find. Focus on content accuracy, including correct capture of formatting elements like tables, lists, and paragraphs."
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url":
                        f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"
                    }
                }]
            }],
            max_tokens=500)

        validation_result = gpt4_response.choices[0].message.content

        # Extract accuracy percentage from GPT-4's response
        accuracy_match = re.search(r'(\d+(?:\.\d+)?)%', validation_result)
        accuracy = float(accuracy_match.group(1)) if accuracy_match else 0.0
        print(f"Accuracy: {accuracy}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_claude_text = claude_text
            best_validation_result = validation_result

        if accuracy >= 85.0:
            return best_claude_text, best_validation_result, accuracy

        print("Accuracy below expectation. Retrying...")

    print(
        f"Max attempts reached. Using best result (Accuracy: {best_accuracy:.2f}%)"
    )
    return best_claude_text, best_validation_result, best_accuracy


def extract_images_from_pdf(pdf_data):
    images = []
    pdf_file = io.BytesIO(pdf_data)
    doc = fitz.open(stream=pdf_file, filetype="pdf")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert to PIL Image for consistency with the rest of your code
            img = Image.open(io.BytesIO(image_bytes))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            images.append(img_byte_arr.getvalue())

    doc.close()
    return images


def text_creation(data, anthropic_client):
    result = ''
    for i, img_data in enumerate(data):
        img = Image.open(io.BytesIO(img_data))

        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_base64 = base64.b64encode(img_data).decode('utf-8')

        claude_response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            messages=[{
                "role":
                "user",
                "content": [{
                    "type":
                    "text",
                    "text":
                    "You are an advanced OCR text extraction tool specialized in processing student answer paper images. Your task is to accurately extract handwritten text and precisely replicate tables from these images. For tables, ensure the correct structure with properly aligned rows and columns, preserving the original layout and content. Do not alter or add any information, as the extracted data will be used for evaluation purposes. Do not generate any intro and outro in the output."
                }, {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_base64
                    }
                }]
            }])
        claude_text = claude_response.content[0].text
        result = result + "\n" + claude_text
    return result


def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()


def compare_ocr_results(claude_text, gemini_text):
    # Preprocess both texts
    claude_words = preprocess_text(claude_text)
    gemini_words = preprocess_text(gemini_text)

    # Count word occurrences
    claude_word_count = Counter(claude_words)
    gemini_word_count = Counter(gemini_words)

    # Find common words
    common_words = set(claude_word_count.keys()) & set(
        gemini_word_count.keys())

    # Calculate similarity based on common words
    total_words = len(set(claude_words + gemini_words))
    similarity = len(
        common_words) / total_words * 100 if total_words > 0 else 0

    # Find differences
    claude_unique = set(claude_words) - set(gemini_words)
    gemini_unique = set(gemini_words) - set(claude_words)

    return {
        'similarity_percentage': similarity,
        'common_words': len(common_words),
        'total_unique_words': total_words,
        'claude_unique_words': list(claude_unique),
        'gemini_unique_words': list(gemini_unique)
    }


def perform_ocr_and_compare(claude_text, gemini_text1, list1,
                            anthropic_client):
    MIN_SIMILARITY = 90.0
    MAX_ATTEMPTS = 1
    attempts = 0
    best_similarity = 0
    best_claude_text = ""
    best_gemini_text = ""
    count = 0
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        print(f"Attempt {attempts} of {MAX_ATTEMPTS}")
        if count == 0:
            comparison_result = compare_ocr_results(claude_text, gemini_text1)
            similarity = comparison_result['similarity_percentage']
        else:
            claude_text_new = text_creation(list1, anthropic_client)
            comparison_result = compare_ocr_results(claude_text_new,
                                                    gemini_text1)
            similarity = comparison_result['similarity_percentage']

        print(f"Similarity: {similarity:.2f}%")
        print(f"Common words: {comparison_result['common_words']}")
        print(f"Total unique words: {comparison_result['total_unique_words']}")

        if similarity > best_similarity:
            best_similarity = similarity
            best_claude_text = claude_text
            best_gemini_text = gemini_text1

        if similarity >= MIN_SIMILARITY:
            print("Acceptable similarity reached.")
            return best_claude_text, best_gemini_text

        print("Similarity below Expectation. Retrying...")
        count = count + 1
        time.sleep(2)  # Add a small delay before retrying

    print(
        f"Max attempts reached. Using best result (Similarity: {best_similarity:.2f}%)"
    )
    return best_claude_text, best_gemini_text


def save_to_database(extracted_info, db_credentials):
    conn = None
    cur = None
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(host=db_credentials['host'],
                                database=db_credentials['database'],
                                user=db_credentials['user'],
                                password=db_credentials['password'],
                                port=db_credentials['port'])

        # Create a cursor
        cur = conn.cursor()

        # Assuming you have a table named 'extracted_data' with columns 'id' and 'data'
        # If the table doesn't exist, create it
        cur.execute("""
            CREATE TABLE IF NOT EXISTS extracted_data (
                id SERIAL PRIMARY KEY,
                data JSONB
            )
        """)

        # Insert the data
        cur.execute("INSERT INTO extracted_data (data) VALUES (%s)",
                    (Json(json.loads(extracted_info)), ))

        conn.commit()

        print("Data successfully saved to database")

    except psycopg2.Error as error:
        print("Error while connecting to PostgreSQL or executing query:",
              error)

    except json.JSONDecodeError:
        print("Error: The extracted_info is not a valid JSON string")

    finally:
        # Close database connection
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("PostgreSQL connection is closed")


def final_process1(claude_text, anthropic_client, db_credentials,
                   final_process_prompt):

    claude_response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        system=
        "You are a precise data extraction assistant. Your task is to extract information from the given text and format it as a valid Python dictionary.",
        messages=[{
            "role":
            "user",
            "content":
            f"{final_process_prompt}\n\nHere is the OCR output: {claude_text}"
        }],
        max_tokens=2000)

    extracted_info = claude_response.content[0].text
    json_match = re.search(r'\{[\s\S]*\}', extracted_info)
    if json_match:
        json_string = json_match.group(0)
        try:
            json_data = json.loads(json_string)
            extracted_info_new = json.dumps(json_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"JSON string: {json_string}")
            return json_string
    else:
        print("No valid JSON found in the extracted info")
        return extracted_info
    save_to_database(extracted_info_new, db_credentials)
    return extracted_info


def final_process2(gpt4_text, client, db_credentials, final_process_prompt):

    gpt4_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role":
            "system",
            "content":
            "You are a precise data extraction assistant. Your task is to extract information from the given text and format it as a valid Python dictionary."
        }, {
            "role":
            "user",
            "content":
            f"{final_process_prompt}\n\nHere is the OCR output: {gpt4_text}"
        }],
        max_tokens=2000)

    extracted_info = gpt4_response.choices[0].message.content
    json_match = re.search(r'\{[\s\S]*\}', extracted_info)
    if json_match:
        json_string = json_match.group(0)
        try:
            json_data = json.loads(json_string)
            extracted_info_new = json.dumps(json_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"JSON string: {json_string}")
            return json_string
    else:
        print("No valid JSON found in the extracted info")
        return extracted_info
    save_to_database(extracted_info_new, db_credentials)
    return extracted_info


def final_process3(claude_text, model, db_credentials, final_process_prompt):

    prompt = f"{final_process_prompt}\n\nHere is the OCR output: {claude_text}"
    gemini_response = model.generate_content(prompt)

    extracted_info = gemini_response.text
    json_match = re.search(r'\{[\s\S]*\}', extracted_info)
    if json_match:
        json_string = json_match.group(0)
        try:
            json_data = json.loads(json_string)
            extracted_info_new = json.dumps(json_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"JSON string: {json_string}")
            return json_string
    else:
        print("No valid JSON found in the extracted info")
        return extracted_info
    save_to_database(extracted_info_new, db_credentials)
    return extracted_info


def final_process4(claude_text, anthropic_client, db_credentials,
                   final_process_prompt):

    claude_response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        system=
        "You are a precise data extraction assistant. Your task is to extract information from the given text and format it as a valid Python dictionary.",
        messages=[{
            "role":
            "user",
            "content":
            f"{final_process_prompt}\n\nHere is the OCR output: {claude_text}"
        }],
        max_tokens=2000)

    extracted_info = claude_response.content[0].text
    json_match = re.search(r'\{[\s\S]*\}', extracted_info)
    if json_match:
        json_string = json_match.group(0)
        try:
            json_data = json.loads(json_string)
            extracted_info_new = json.dumps(json_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"JSON string: {json_string}")
            return json_string
    else:
        print("No valid JSON found in the extracted info")
        return extracted_info
    save_to_database(extracted_info_new, db_credentials)
    return extracted_info


def final_process5(gpt4_text, client, db_credentials, final_process_prompt):

    gpt4_response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{
            "role":
            "system",
            "content":
            "You are a precise data extraction assistant. Your task is to extract information from the given text and format it as a valid Python dictionary."
        }, {
            "role":
            "user",
            "content":
            f"{final_process_prompt}\n\nHere is the OCR output: {gpt4_text}"
        }],
        max_tokens=2000)

    extracted_info = gpt4_response.choices[0].message.content
    json_match = re.search(r'\{[\s\S]*\}', extracted_info)
    if json_match:
        json_string = json_match.group(0)
        try:
            json_data = json.loads(json_string)
            extracted_info_new = json.dumps(json_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"JSON string: {json_string}")
            return json_string
    else:
        print("No valid JSON found in the extracted info")
        return extracted_info
    save_to_database(extracted_info_new, db_credentials)
    return extracted_info


def final_process6(claude_text, model, db_credentials, final_process_prompt):

    prompt = f"{final_process_prompt}\n\nHere is the OCR output: {claude_text}"
    gemini_response = model.generate_content(prompt)

    extracted_info = gemini_response.text
    json_match = re.search(r'\{[\s\S]*\}', extracted_info)
    if json_match:
        json_string = json_match.group(0)
        try:
            json_data = json.loads(json_string)
            extracted_info_new = json.dumps(json_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"JSON string: {json_string}")
            return json_string
    else:
        print("No valid JSON found in the extracted info")
        return extracted_info
    save_to_database(extracted_info_new, db_credentials)
    return extracted_info


def final_process7(claude_text, anthropic_client, db_credentials,
                   final_process_prompt):

    claude_response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        system=
        "You are a precise data extraction assistant. Your task is to extract information from the given text and format it as a valid Python dictionary.",
        messages=[{
            "role":
            "user",
            "content":
            f"{final_process_prompt}\n\nHere is the OCR output: {claude_text}"
        }],
        max_tokens=2000)

    extracted_info = claude_response.content[0].text
    json_match = re.search(r'\{[\s\S]*\}', extracted_info)
    if json_match:
        json_string = json_match.group(0)
        try:
            json_data = json.loads(json_string)
            extracted_info_new = json.dumps(json_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"JSON string: {json_string}")
            return json_string
    else:
        print("No valid JSON found in the extracted info")
        return extracted_info
    save_to_database(extracted_info_new, db_credentials)
    return extracted_info


def json_to_html_with_claude(json_string, anthropic_client):

    prompt = f"""
    You are an expert in converting JSON data to semantic HTML. Your task is to take the following JSON data and convert it into a well-structured, accessible HTML format. Use appropriate HTML tags to represent the data structure and content meaningfully.

    Here's the JSON data:
    {json_string}

    Please convert this JSON into HTML with the following guidelines:
    1. Use semantic HTML5 tags where appropriate (e.g., <header>, <main>, <section>, <article>, <table> if it's tabular data, etc.).
    2. Add appropriate classes or ids to elements for potential styling.
    3. Ensure the HTML is valid and well-formatted.
    4. Use lists (<ul> or <ol>) for array data if appropriate.
    5. Add a simple, clean CSS style to make the output visually organized (include this within a <style> tag).

    Provide only the HTML output, without any explanations or markdown formatting.
    """

    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": prompt
        }])

    html_output = response.content[0].text
    return html_output


'''
@csrf_exempt
@api_view(['POST', 'GET'])
def Value1(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        option1 = request.POST.get('validation_model')
        option2 = request.POST.get('refining_model')
        validation_prompt = request.POST.get('validation_prompt')
        refining_prompt = request.POST.get('refining_prompt')
        print(file.content_type)
        file_data = file.read()

        if file.content_type == 'application/pdf':
            images = convert_from_bytes(file_data)
            image_data = [io.BytesIO() for _ in images]
            for img, img_data in zip(images, image_data):
                img.save(img_data, format='JPEG')
        elif file.content_type in ['image/png', 'image/jpeg']:
            image_data = [file_data]
        else:
            return JsonResponse({'error': 'Unsupported file type'}, status=400)

        anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_KEY'))
        client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
        genai.configure(api_key=os.getenv('Gemnai'))
        gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        gemini_flash_model = genai.GenerativeModel('gemini-1.5-flash')
        db_url = os.getenv('DATABASE_URL')
        result = urllib.parse.urlparse(db_url)
        db_credentials = {
            'host': result.hostname,
            'database': result.path[1:],
            'user': result.username,
            'password': result.password,
            'port': result.port
        }

        helper_images = []
        main_text = ''
        for img_data in image_data:

            if option1 == 'GEMINI 1.5 PRO':
                claude_text, validation_result, accuracy = perform_ocr_and_validate3(
                    img_data, gemini_model, client, validation_prompt)
            elif option1 == 'CLAUDE 3.5 SONNET':
                claude_text, validation_result, accuracy = perform_ocr_and_validate1(
                    img_data, anthropic_client, client, validation_prompt)
            elif option1 == 'GPT 4o':
                claude_text, validation_result, accuracy = perform_ocr_and_validate2(
                    img_data, anthropic_client, client, validation_prompt)
            elif option1 == 'CLAUDE 3 HAIKU':
                claude_text, validation_result, accuracy = perform_ocr_and_validate4(
                    img_data, anthropic_client, client, validation_prompt)
            elif option1 == 'GEMINI 1.5 FLASH':
                claude_text, validation_result, accuracy = perform_ocr_and_validate6(
                    img_data, gemini_flash_model, client, validation_prompt)
            else:  # CLAUDE OPUS
                claude_text, validation_result, accuracy = perform_ocr_and_validate7(
                    img_data, anthropic_client, client, validation_prompt)

            print(f"Accuracy: {accuracy}%")
            main_text += claude_text
            helper_images.append(img_data)

        claude_text = main_text
        print("Main text: done .................")
        # Final processing
        if option2 == 'CLAUDE 3.5 SONNET':
            extracted_info = final_process1(claude_text, anthropic_client,
                                            db_credentials, refining_prompt)
        elif option2 == 'GEMINI 1.5 PRO':
            extracted_info = final_process3(claude_text, gemini_model,
                                            db_credentials, refining_prompt)
        elif option2 == 'GPT 4o':
            extracted_info = final_process2(claude_text, client,
                                            db_credentials, refining_prompt)
        elif option2 == 'CLAUDE 3 HAIKU':
            extracted_info = final_process4(claude_text, anthropic_client,
                                            db_credentials, refining_prompt)
        elif option2 == 'GEMINI 1.5 FLASH':
            extracted_info = final_process6(claude_text, gemini_flash_model,
                                            db_credentials, refining_prompt)
        elif option2 == 'GPT 3.5':
            extracted_info = final_process5(claude_text, client,
                                            db_credentials, refining_prompt)
        else:  # CLAUDE OPUS
            extracted_info = final_process7(claude_text, anthropic_client,
                                            db_credentials, refining_prompt)
        print("refine done")

        html_output = json_to_html_with_claude(extracted_info,
                                               anthropic_client)
        print("html done")
        response_data = {
            'main_text': main_text,
            'extracted_info': extracted_info,
            'html_output': html_output
        }

        # Return the JsonResponse
        return JsonResponse(response_data, safe=False)

        # If the request method is GET, you might want to return a different response
    elif request.method == 'GET':
        return JsonResponse(
            {'message': 'Please use POST method to process files'})

    # If neither POST nor GET, return an error
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
'''

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@csrf_exempt
@api_view(['POST', 'GET'])
def Value1(request):
    try:
        if request.method == 'POST':
            file = request.FILES.get('file')
            option1 = request.POST.get('validation_model')
            option2 = request.POST.get('refining_model')
            validation_prompt = request.POST.get('validation_prompt')
            refining_prompt = request.POST.get('refining_prompt')
            print(file.content_type)
            print(option1)
            print(option2)
            print(validation_prompt)
            print(refining_prompt)

            if not file or not option1 or not option2 or not validation_prompt or not refining_prompt:
                return JsonResponse({'error': 'Missing required parameters'},
                                    status=400)

            file_data = file.read()
            image_data = []
            if file.content_type == 'application/pdf':
                images = convert_from_bytes(file_data)
                for img in images:
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    image_data.append(img_byte_arr.getvalue())
            elif file.content_type in ['image/png', 'image/jpeg']:
                image_data = [file_data]
            else:
                return JsonResponse({'error': 'Unsupported file type'},
                                    status=400)

            anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_KEY'))
            openai_client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
            genai.configure(api_key=os.getenv('GENAI_KEY'))
            gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            gemini_flash_model = genai.GenerativeModel('gemini-1.5-flash')
            db_url = os.getenv('DATABASE_URL')
            result = urllib.parse.urlparse(db_url)
            db_credentials = {
                'host': result.hostname,
                'database': result.path[1:],
                'user': result.username,
                'password': result.password,
                'port': result.port
            }

            helper_images = []
            main_text = ''
            for img_data in image_data:
                if option1 == 'GEMINI 1.5 PRO':
                    claude_text, validation_result, accuracy = perform_ocr_and_validate3(
                        img_data, gemini_model, openai_client,
                        validation_prompt)
                elif option1 == 'CLAUDE 3.5 SONNET':
                    claude_text, validation_result, accuracy = perform_ocr_and_validate1(
                        img_data, anthropic_client, openai_client,
                        validation_prompt)
                elif option1 == 'GPT 4o':
                    claude_text, validation_result, accuracy = perform_ocr_and_validate2(
                        img_data, anthropic_client, openai_client,
                        validation_prompt)
                elif option1 == 'CLAUDE 3 HAIKU':
                    claude_text, validation_result, accuracy = perform_ocr_and_validate4(
                        img_data, anthropic_client, openai_client,
                        validation_prompt)
                elif option1 == 'GEMINI 1.5 FLASH':
                    claude_text, validation_result, accuracy = perform_ocr_and_validate6(
                        img_data, gemini_flash_model, openai_client,
                        validation_prompt)
                else:  # CLAUDE OPUS
                    claude_text, validation_result, accuracy = perform_ocr_and_validate7(
                        img_data, anthropic_client, openai_client,
                        validation_prompt)

                logger.debug(f"Accuracy: {accuracy}%")
                main_text += claude_text

                helper_images.append(img_data)

            logger.debug("Main text processing done")

            if option2 == 'CLAUDE 3.5 SONNET':
                extracted_info = final_process1(claude_text, anthropic_client,
                                                db_credentials,
                                                refining_prompt)
            elif option2 == 'GEMINI 1.5 PRO':
                extracted_info = final_process3(claude_text, gemini_model,
                                                db_credentials,
                                                refining_prompt)
            elif option2 == 'GPT 4o':
                extracted_info = final_process2(claude_text, openai_client,
                                                db_credentials,
                                                refining_prompt)
            elif option2 == 'CLAUDE 3 HAIKU':
                extracted_info = final_process4(claude_text, anthropic_client,
                                                db_credentials,
                                                refining_prompt)
            elif option2 == 'GEMINI 1.5 FLASH':
                extracted_info = final_process6(claude_text,
                                                gemini_flash_model,
                                                db_credentials,
                                                refining_prompt)
            elif option2 == 'GPT 3.5':
                extracted_info = final_process5(claude_text, openai_client,
                                                db_credentials,
                                                refining_prompt)
            else:  # CLAUDE OPUS
                extracted_info = final_process7(claude_text, anthropic_client,
                                                db_credentials,
                                                refining_prompt)

            logger.debug("Refinement processing done")

            html_output = json_to_html_with_claude(extracted_info,
                                                   anthropic_client)
            logger.debug("HTML generation done")

            response_data = {
                'main_text': main_text,
                'extracted_info': extracted_info,
                'html_output': html_output
            }
            return JsonResponse(response_data, safe=False)

        elif request.method == 'GET':
            return JsonResponse(
                {'message': 'Please use POST method to process files'})

        else:
            return JsonResponse({'error': 'Invalid request method'},
                                status=405)

    except Exception as e:
        logger.error(f"Error in Value1 view: {str(e)}")
        return JsonResponse(
            {
                'error': 'Internal Server Error',
                'details': str(e)
            }, status=500)
