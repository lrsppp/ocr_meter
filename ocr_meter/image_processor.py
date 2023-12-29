from typing import Any, Dict, List, Tuple, Union


class ImageProcessor:
    def __init__(self, pipeline: List[Tuple[callable, Dict[str, Any]]]) -> None:
        """
        Initialize the ImageProcessor with a list of processing pipeline.

        Example:
            >>> pipeline = [
                    (cv2.imread, {}),
                    (cv2.cvtColor, {"code": cv2.COLOR_BGR2GRAY}),
                    (cv2.adaptiveThreshold, {"maxValue": 255, "adaptiveMethod": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}),
                ]
            >>> proc = ImageProcessor(pipeline)
            >>> img_processed = proc.process("/path/to/img.png")

        :param pipeline: List of tuples where each tuple contains a callable function and its parameters.
        """
        self.pipeline = pipeline

    def process(
        self, image: Union[None, List[List[Any]], Any]
    ) -> List[Union[None, List[List[Any]], Any]]:
        """
        Process the input image through all defined pipelines.

        :param image: The input image to be processed.
        :return: List of processed images for each pipeline.
        """
        for step_func, step_params in self.pipeline:
            image = step_func(image, **step_params)
        return image
