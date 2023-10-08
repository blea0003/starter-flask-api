from extraction import get_boxes, display
import cv2
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import io

endpoint = "https://mds12htr.cognitiveservices.azure.com/"
region = "southeastasia"
key = "59084e0384ae403fa0fbc957fc9f9f20"

credentials = CognitiveServicesCredentials(key)
client = ComputerVisionClient(
    endpoint="https://" + region + ".api.cognitive.microsoft.com/",
    credentials=credentials
)


def text_box_htr(doc_img, model="keras_ocr", debug=False):
    """
    Returns detected texts in boxes sequentially
    """

    text_boxes = get_boxes(doc_img, debug=debug)
    recognized_texts = []

    if model == "keras_ocr":
    
        try:
            for img in text_boxes:
                _, image_data = cv2.imencode(".jpg", img)
                image_bytes = image_data.tobytes()

                image_stream = io.BytesIO(image_bytes)

                language = "en"
                result = client.recognize_printed_text_in_stream(image_stream, language=language)

                if result:
                    recognized_words = []
                    for region in result.regions:
                        for line in region.lines:
                            recognized_words.extend(line.words)

                    pred = ""
                    for word in recognized_words:
                        pred += str(word.text)

                else:
                    pred = ""

                recognized_texts.append(pred)
        except:
            raise Exception("Hosting service tier limit exceeded. Try again in a minute")
    
    return recognized_texts 



def patient_registration_parser(img, model="keras_ocr", debug=False):

    recognized_texts = text_box_htr(img, model=model, debug=debug)

    # If not all text boxes detected, just return empty values for all
    if len(recognized_texts) != 11:
        recognized_texts = [''] * 11

    dict = {
        "ic": recognized_texts[0],
        "firstName": recognized_texts[1],
        "lastName": recognized_texts[2],
        "dob": recognized_texts[3],
        "gender": recognized_texts[4],
        "nationality": recognized_texts[5],
        "phoneNo": recognized_texts[6],
        "email": recognized_texts[7],
        "emergencyNo": recognized_texts[8],
        "emergencyRemarks": recognized_texts[9],
        "remarks": recognized_texts[10]
    }

    return dict


if __name__ == "__main__":
    img = cv2.imread("docs/test/digital_1.jpg")
    output = patient_registration_parser(img, model="keras_ocr", debug=False)
    print(output)
