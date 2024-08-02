from flask import Flask, jsonify, request

import cv2
import face_recognition
import urllib.request
import numpy as np
from PIL import Image

import requests
import base64
import io
import json

from fdk import response

def handler(ctx, data: io.BytesIO=None):
    try:
        body = json.loads(data.getvalue())
        image_base64 = body.get("image")
    except (Exception, ValueError) as ex:
        print(str(ex), flush=True)

    try:
        known_face_encodings = np.load('known_face_encodings.npy')
    except Exception as loadFileError:
        return jsonify({"loadEncodingsFileError": str(loadFileError)}), 400

    try:
        with open("known_face_names.txt", 'r') as f:
            known_face_names = f.read().splitlines()
    except Exception as loadFileError:
        return jsonify({"loadNamesFileError": str(loadFileError)}), 400

    try:
        image_data = base64.b64decode(image_base64)
    except base64.binascii.Error:
        return jsonify({"Base64DecodeError": "Invalid base64 string"}), 400

    try:
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
    except Exception as e:
        return jsonify({"ImageOpenError": str(e)}), 400

    try:
        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)
    except Exception as e:
        return jsonify({"FaceEncodingError": str(e)}), 400

    try:
        face_names = []
        for face_encoding in face_encodings:
            try:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
            except Exception as e:
                return jsonify({"CompareFacesError": str(e)}), 400

            try:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
            except Exception as e:
                return jsonify({"FaceDistanceError": str(e)}), 400

        face_locations = np.array(face_locations)
        response = jsonify({"face_ids": face_names, "face_locations": face_locations.astype(int).tolist()})
        return response
    except Exception as genErr:
        return jsonify({"GeneralError": str(genErr)}), 400
