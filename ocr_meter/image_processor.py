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


# class ImageEvaluator:
#     def __init__(self, pipelines: List[List[Tuple[callable, Dict[str, Any]]]]):
#         self.pipelines = pipelines
#         self.processors = [ImageProcessor(pipeline) for pipeline in self.pipelines]

#     def evaluate(self, X: List[Any], y: List[Any]):
#         results = []
#         for processor in self.processors:
#             y_pred = []
#             for x_ in X:
#                 y_pred.append(processor.process(x_))

#             results.append((y_pred, X, y, processor.pipeline))
#         return results


# def create_pipeline_grid(pipeline: List[Tuple[callable, Dict[str, Any]]]):
#     pass
