from extraction import get_boxes, display
import cv2
import tensorflow as tf
import keras_ocr

def text_box_htr(doc_img, model="keras_ocr", debug=False):
    """
    Returns detected texts in boxes sequentially
    """

    text_boxes = get_boxes(doc_img, debug=debug)
    recognized_texts = []

    if model == "keras_ocr":
        pipeline = keras_ocr.pipeline.Pipeline()
    
        for img in text_boxes:
            pred = pipeline.recognize([img])

            if pred[0] == []:
                pred = ""

            else:
                pred = pred[0][0][0]
            recognized_texts.append(pred)

    if model == "self_implementation":
        # Load model
        model = tf.keras.models.load_model('model\models\htr_model')
        prediction_model = tf.keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)

        for img in text_boxes:
            # Preprocess image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Greyscale
            img = 255 - img                             # Invert
            img = cv2.flip(img, 1)                      # Flip
            img = cv2.transpose(img)                    # Transpose
            img = img / 255                             # Normalize
            img = cv2.resize(img, (30, 200))            # Scale down
            img = tf.expand_dims(img, axis=0)
            img = tf.expand_dims(img, axis=3)

            # Predict text in image
            pred = prediction_model.predict(img)

            # Decode prediction (halted progress)

            # Append prediction
            # recognized_texts.append(pred)

    
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