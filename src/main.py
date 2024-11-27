import asyncio
import sys
from typing import ClassVar, List, Mapping, Optional, Any, cast
from typing_extensions import Self

from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import (Classification, Detection,
                                       GetPropertiesResponse)
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import *
from viam.components.camera import Camera, ViamImage
from viam.utils import ValueTypes
from viam.logging import getLogger
from viam.media.utils.pil import viam_to_pil_image

from PIL import Image
import numpy as np
import cv2
import apriltag
import time

LOGGER = getLogger(__name__)

class Apriltag(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("joyce", "vision"), "apriltag")

    last_triggered: dict = {}

    cooldown_period: int = 5
    
    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        my_class = cls(config.name)
        my_class.reconfigure(config, dependencies)
        return my_class

    # Validates JSON Configuration
    @classmethod
    def validate(cls, config: ComponentConfig):
        return

    # Handles attribute reconfiguration
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        self.DEPS = dependencies
        return
        
    async def get_cam_image(self, camera_name: str) -> ViamImage:

        LOGGER.info(f"Fetching image from camera: {camera_name}")

        if Camera.get_resource_name(camera_name) not in self.DEPS:
            LOGGER.error(f"Camera {camera_name} is not available in dependencies")
            raise ValueError(f"Camera {camera_name} not found")

        actual_cam = self.DEPS[Camera.get_resource_name(camera_name)]
        cam = cast(Camera, actual_cam)
        cam_image = await cam.get_image(mime_type="image/jpeg")

        LOGGER.debug(f"Image fetched successfully from camera: {camera_name}")

        return cam_image

    async def get_detections_from_camera(self, camera_name: str, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None) -> List[Detection]:
        # Get image from the camera
        cam_image = await self.get_cam_image(camera_name)
        return await self.detect_april_tag(cam_image)

    async def get_detections(
        self,
        image: Image.Image,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        # Convert PIL image to ViamImage or OpenCV as needed
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        return self.detect_april_tag(ViamImage(data=image_cv.tobytes(), mime_type="image/jpeg"))

    async def get_classifications(
        self,
        image: Image.Image,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        """
        This method is not implemented for Apriltag detection.
        """
        LOGGER.warning("get_classifications is not implemented for AprilTag detection")
        return []

    async def detect_april_tag(self, image: ViamImage) -> List[Detection]:
        """
        Detect April Tags in the given image using apriltag.
        """
        try:
            # Convert ViamImage to OpenCV format
            image_pil = viam_to_pil_image(image)
            image_cv = np.array(image_pil)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale for apriltag

            # Initialize color image for visualization (always)
            image_cv_color = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)

            # Initialize AprilTag detector
            # Can include multiple families of tags in comma separated string
            options = apriltag.DetectorOptions(families="tag36h11")
            detector = apriltag.Detector(options)
            tags = detector.detect(image_cv)
            detections = []

            # If tags are detected, process and visualize them
            if not tags:
                LOGGER.info("No April Tags detected")
                return detections
            for tag in tags:
                tag_data = tag.tag_id
                LOGGER.info(f"AprilTag detected: {tag_data}")

                # Draw bounding boxes on the image
                corners = np.array(tag.corners, dtype=np.int32)
                cv2.polylines(image_cv_color, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

                # Display tag ID at the center of the tag
                center = (int(tag.center[0]), int(tag.center[1]))
                cv2.putText(
                    image_cv_color, f"ID: {tag_data}", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                # Create a Detection object for each tag detected
                x_min = int(min(c[0] for c in corners))
                y_min = int(min(c[1] for c in corners))
                x_max = int(max(c[0] for c in corners))
                y_max = int(max(c[1] for c in corners))
                detection = Detection(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    class_name=str(tag_data),
                    confidence=1.0  # Assuming full confidence
                )
                detections.append(detection)

            # Show the processed image with detected tags
            # cv2.imshow("Detected AprilTags", image_cv_color)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()

            return detections
        except Exception as e:
            LOGGER.error(f"Error during AprilTag detection: {e}", exc_info=True)
            return []
    
    def preprocess_image(self, image):
        """
        Preprocess the image to improve Apriltag detection.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)
        threshold_image = cv2.threshold(equalized_image, 128, 255, cv2.THRESH_BINARY)[1]
        resized_image = cv2.resize(threshold_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        return resized_image
    
    def trigger_action_on_april_tag(self, april_tag_data: str):
        """
        Trigger an action based on the AprilTag data.
        """
        
        # Can add custom logic here
        LOGGER.info(f"Triggering action based on April Tag: {april_tag_data}")

    async def get_classifications_from_camera(self, camera_name: str, count: int, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None) -> List[Classification]:
        """
        This method is not implemented for Apriltag detection.
        """
        LOGGER.warning("get_classifications_from_camera is not implemented for AprilTag detection")
        return []
    
    async def get_object_point_clouds(self, camera_name: str, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None) -> List[PointCloudObject]:
        return []
    
    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None) -> Mapping[str, ValueTypes]:
        return {}

    async def capture_all_from_camera(self, camera_name: str, return_image: bool = False, return_classifications: bool = False, return_detections: bool = False, return_object_point_clouds: bool = False, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None) -> CaptureAllResult:
        result = CaptureAllResult()
        result.image = await self.get_cam_image(camera_name)
        result.detections = await self.get_detections_from_camera(camera_name)
        return result

    async def get_properties(self, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False
        )

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
