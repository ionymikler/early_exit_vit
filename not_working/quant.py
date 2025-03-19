#!/usr/bin/env python
# Made by: Jonathan Mikler on 2025-03-13

import os
from PIL import Image
import numpy as np
import time
from onnxruntime.quantization import (
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationDataReader,
)
import onnxruntime as ort

import onnx
from onnxconverter_common import float16


def benchmark(model_path):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 224, 224), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        input_data = np.float32(pillow_img) - np.array(
            [123.68, 116.78, 103.94], dtype=np.float32
        )
        nhwc_data = np.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(
        np.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class Cifar100DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = ort.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, height, width, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def main():
    input_model_path = "models/onnx/EEVIT.onnx"
    output_model_path = "models/onnx/EEVIT_int8.onnx"
    calibrate_dataset_path = "./results/cifar100_images"

    model = onnx.load(input_model_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, "models/onnx/EEVIT_fp16.onnx")

    dr = Cifar100DataReader(calibrate_dataset_path, input_model_path)

    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
    )

    benchmark(input_model_path)
    benchmark(output_model_path)


if __name__ == "__main__":
    main()
