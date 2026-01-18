from typing import Any
from PIL import Image
from comfy_api.latest import ComfyExtension, io as comfy_api_io # pyright: ignore[reportMissingImports]
import base64
import io
import json
import numpy as np
import time
import websocket

import torch # pyright: ignore[reportMissingImports]
import boto3
import msgpack
import zstd

class PixelSocketPutObjectStorageNode(comfy_api_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_api_io.Schema:
        return comfy_api_io.Schema(
            node_id="PixelSocketPutObjectStorageNode",
            display_name="Pixel Socket Put Object Storage Node",
            category="PixelSocket",
            inputs=[
                comfy_api_io.Image.Input("image"),
                comfy_api_io.String.Input("websocket_url",
                    default="wss://example.foundation0.link/ws/streaming",
                    optional=False
                ),
                comfy_api_io.Combo.Input("file_format",
                    options=["WEBP", "PNG", "JPEG"],
                    default="WEBP"
                ),
                comfy_api_io.String.Input("endpoint_url",
                    default="https://<ACCOUNT_ID>.r2.cloudflarestorage.com",
                    optional=False
                ),
                comfy_api_io.String.Input("bucket_name",
                    default="<BUCKET_NAME>",
                    optional=False
                ),
                comfy_api_io.String.Input("aws_access_key_id",
                    default="<ACCESS_KEY_ID>",
                    optional=False
                ),
                comfy_api_io.String.Input("aws_secret_access_key",
                    default="<SECRET_ACCESS_KEY>",
                    optional=False
                ),
                comfy_api_io.String.Input("secret_token",
                    default="<SECRET_TOKEN>",
                    optional=False
                ),
                comfy_api_io.String.Input("request_job_id",
                    default="<REQUEST_JOB_ID>",
                    optional=False
                ),
            ],
            outputs=[
                comfy_api_io.Image.Output()
            ]
        )

    @classmethod
    def execute(cls, image, file_format, websocket_url, endpoint_url, bucket_name, aws_access_key_id, aws_secret_access_key, secret_token, request_job_id, **kwargs) -> comfy_api_io.NodeOutput:
        try:
            epoch_time:int = int(time.time() * 1000)

            img_bytes = PixelSocketExtensions.tensor_to_image_bytes(image, file_format)
            img_size = len(img_bytes)

            s3_client = boto3.client(
                service_name="s3",
                endpoint_url=endpoint_url,
                # Retrieve your S3 API credentials for your R2 bucket via API tokens (see: https://developers.cloudflare.com/r2/api/tokens)
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name="auto", # Required by SDK but not used by R2
            )

            file_key = f"comfyui_{request_job_id}.{epoch_time}.{file_format.lower()}"

            s3_client.put_object(
                Bucket=bucket_name,
                Key=file_key,
                ContentType=f"image/{file_format.lower()}",
                Body=img_bytes
            )
            del img_bytes

            object_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': file_key},
                ExpiresIn=3600 * 24 * 7  # 7 days
            )

            # Create payload
            payload: dict[str, Any] = {
                "type": "notification-from-pixel-socket",
                "payload": {
                    "jobId": request_job_id,
                    "blobData": None,  # Not sending image data here
                    "imageLength": img_size,
                    "fileExtension": file_format.lower(),
                    "mimeType": f"image/{file_format.lower()}",
                    "objectUrl": object_url,
                    "secretToken": secret_token,
                    "timestamp": epoch_time,
                }
            }
            packed: bytes = msgpack.packb(payload, use_bin_type=True)
            compressed_data: bytes = zstd.compress(packed, 22) # High compression level:22

            ws: websocket.WebSocket | None = None
            try:
                ws = websocket.create_connection(websocket_url)
                ws.send(compressed_data, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as ex:
                print(f"WebSocket error: {ex}")
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                ws = None

        except Exception:
            import traceback
            traceback.print_exc()

        return comfy_api_io.NodeOutput(image)

class PixelSocketDeliveryImageNode(comfy_api_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_api_io.Schema:
        return comfy_api_io.Schema(
            node_id="PixelSocketDeliveryImageNode",
            display_name="Pixel Socket Delivery Image Node",
            category="PixelSocket",
            inputs=[
                comfy_api_io.Image.Input("image"),
                comfy_api_io.String.Input("websocket_url",
                    default="wss://example.foundation0.link/ws/streaming",
                    optional=False
                ),
                comfy_api_io.Combo.Input("file_format",
                    options=["WEBP", "PNG", "JPEG"],
                    default="WEBP"
                ),
                comfy_api_io.String.Input("secret_token",
                    default="generate_random_token",
                    optional=False
                ),
                comfy_api_io.String.Input("request_job_id",
                    default="<REQUEST_JOB_ID>",
                    optional=False
                ),
                comfy_api_io.String.Input("checkpoint_name",
                    default="",
                    optional=True,
                    multiline=False
                ),
                comfy_api_io.String.Input("positive_prompt",
                    default="",
                    optional=True,
                    multiline=True
                ),
                comfy_api_io.String.Input("negative_prompt",
                    default="",
                    optional=True,
                    multiline=True
                ),
                comfy_api_io.Int.Input("seed_value",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    step=1,
                    optional=True,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
                comfy_api_io.Int.Input("width",
                    default=512,
                    min=1,
                    max=8192,
                    step=8,
                    optional=True,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
                comfy_api_io.Int.Input("height",
                    default=512,
                    min=1,
                    max=8192,
                    step=8,
                    optional=True,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
                comfy_api_io.Int.Input("step",
                    default=20,
                    min=1,
                    max=100,
                    step=1,
                    optional=True,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
                comfy_api_io.Float.Input("cfg",
                    default=8.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    optional=True,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
            ],
            outputs=[
            ]
        )

    @classmethod
    def execute(cls, image: torch.Tensor, websocket_url, file_format, secret_token, request_job_id, **kwargs) -> None:
        try:
            epoch_time:int = int(time.time() * 1000)

            img_bytes = PixelSocketExtensions.tensor_to_image_bytes(image, file_format)
            img_size = len(img_bytes)

            # Create payload
            payload: dict[str, Any] = {
                "type": "notification-from-pixel-socket",
                "payload": {
                    "jobId": request_job_id,
                    "blobData": img_bytes,
                    "imageLength": img_size,
                    "fileExtension": file_format.lower(),
                    "mimeType": f"image/{file_format.lower()}",
                    "objectUrl": None,
                    "secretToken": secret_token,
                    "timestamp": epoch_time,
                }
            }
            packed: bytes = msgpack.packb(payload, use_bin_type=True)
            compressed_data: bytes = zstd.compress(packed, 22) # High compression level:22

            ws: websocket.WebSocket | None = None
            try:
                ws = websocket.create_connection(websocket_url)
                ws.send(compressed_data, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as ex:
                print(f"WebSocket error: {ex}")
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                ws = None

        except Exception:
            import traceback
            traceback.print_exc()

class PixelSocketExtensions(ComfyExtension):
    async def get_node_list(self) -> list[type[comfy_api_io.ComfyNode]]:
        return [PixelSocketPutObjectStorageNode, PixelSocketDeliveryImageNode]

    @classmethod
    def tensor_to_image_bytes(cls, image: torch.Tensor, file_format: str) -> bytes:
        arr = image.detach().cpu().numpy()

        # 余分な次元を削除
        while arr.ndim > 3:
            arr = arr[0]

        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[-1] not in (3, 4):
            raise ValueError(f"Unsupported channel count: {arr.shape}")

        img = Image.fromarray(arr)

        buf = io.BytesIO()
        if file_format.lower() == "png":
            img.save(buf, format="PNG")
        elif file_format.lower() == "webp":
            img.save(buf, format="WEBP", quality=90, method=6)
        elif file_format.lower() in ("jpeg", "jpg"):
            img.save(buf, format="JPEG", quality=90)
        else:
            raise ValueError("Unsupported format")

        return buf.getvalue()

async def comfy_entrypoint() -> ComfyExtension:
    return PixelSocketExtensions()
