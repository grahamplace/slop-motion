import io

import pytest
from PIL import Image, ImageDraw

from stop_motion.images import GeneratedImage
from stop_motion.project import Project


def png_bytes(color="white", size=(1024, 1024), step=0):
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.ellipse((50 + step * 10, 50, 100 + step * 10, 100), fill="#ffa500")
    result = io.BytesIO()
    image.save(result, format="PNG")
    return result.getvalue()


class FakeImages:
    def __init__(self, colors=("#c83232", "#32c832"), size=(1024, 1024)):
        self.calls = []
        self.colors = colors
        self.size = size

    def generate(self, request):
        self.calls.append((request, request.input_paths))
        index = len(self.calls) - 1
        return GeneratedImage(
            png_bytes(self.colors[index % len(self.colors)], self.size, index),
            usage={"total_tokens": 100},
            request_id=f"req_test_{index}",
            metadata={"response_id": f"resp_test_{index}"},
            continuation={"response_id": f"resp_test_{index}"}
            if request.settings["provider_options"]["backend"] == "responses"
            else None,
        )


@pytest.fixture
def project(tmp_path):
    return Project.create(
        tmp_path / "shot with spaces", "A clay butterfly lifts off.", backend="images"
    )


@pytest.fixture
def generator():
    return FakeImages()
