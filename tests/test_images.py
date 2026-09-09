import base64

import httpx
import pytest
from openai import OpenAI

from stop_motion.images import ImageRequestError, OpenAIImages

from .conftest import png_bytes


def sdk_client(handler):
    return OpenAI(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_generate_and_edit_use_real_sdk_with_correct_routes_and_files(tmp_path):
    png = png_bytes()
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200,
            headers={"x-request-id": "req_local"},
            json={
                "created": 0,
                "data": [{"b64_json": base64.b64encode(png).decode()}],
                "usage": {"total_tokens": 100, "input_tokens": 20, "output_tokens": 80},
            },
        )

    with sdk_client(respond) as client:
        images = OpenAIImages(client)
        options = {
            "model": "gpt-image-2.5-sunburst",
            "prompt": "A butterfly",
            "size": "1024x1024",
            "quality": "medium",
            "output_format": "png",
            "background": "opaque",
            "n": 1,
        }
        generated = images.generate(options, [])
        assert generated.png == png
        assert generated.request_id == "req_local"
        assert generated.usage["total_tokens"] == 100
        first, second = tmp_path / "base.png", tmp_path / "reference.png"
        first.write_bytes(png)
        second.write_bytes(png_bytes("blue"))
        edited = images.generate(options, [first, second])
        assert edited.png == png
    assert [request.url.path for request in calls] == ["/v1/images/generations", "/v1/images/edits"]
    assert b'"model":"gpt-image-2.5-sunburst"' in calls[0].content
    multipart = calls[1].content
    assert multipart.index(b'filename="base.png"') < multipart.index(b'filename="reference.png"')
    assert png in multipart


@pytest.mark.parametrize("status", [429, 500])
def test_sdk_never_retries_failed_requests(status):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            status,
            headers={"x-request-id": "req_failure"},
            json={
                "error": {"message": "Unavailable", "type": "api_error", "code": "unavailable"},
            },
        )

    with sdk_client(respond) as client, pytest.raises(ImageRequestError) as error:
        OpenAIImages(client).generate({"model": "gpt-image-2.5-sunburst", "prompt": "Scene"}, [])
    assert len(calls) == 1
    assert error.value.request_id == "req_failure"
    assert error.value.unknown == (status == 500)


def test_timeout_has_unknown_outcome_and_no_retry():
    calls = []

    def timeout(request):
        calls.append(request)
        raise httpx.ReadTimeout("Timed out", request=request)

    with sdk_client(timeout) as client, pytest.raises(ImageRequestError) as error:
        OpenAIImages(client).generate({"model": "gpt-image-2.5-sunburst", "prompt": "Scene"}, [])
    assert error.value.unknown
    assert len(calls) == 1
