from typing import Any
import time
import websocket

import comfy # pyright: ignore[reportMissingImports]
from comfy_api.latest import io as comfy_api_io # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]
import msgpack
import zstd

from .pixel_socket_utils import PixelSocketUtils

class PixelSocketDeliveryImageNode(comfy_api_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_api_io.Schema:
        return comfy_api_io.Schema(
            node_id="PixelSocketDeliveryImageNode",
            display_name="Delivery Image Node",
            category="PixelSocket/Delivery",
            is_output_node=True,
            inputs=[
                comfy_api_io.Image.Input("image"),
                comfy_api_io.Combo.Input("file_format",
                    options=["PNG", "WEBP"],
                    default="PNG"
                ),
                comfy_api_io.String.Input("websocket_url",
                    default="wss://example.foundation0.link/ws/streaming",
                    optional=False
                ),
                comfy_api_io.String.Input("secret_token",
                    default="generate_random_token",
                    optional=False
                ),
                comfy_api_io.String.Input("request_job_id",
                    default="<REQUEST_JOB_ID>",
                    optional=False
                ),
                comfy_api_io.Int.Input("oxipng_level",
                    default=0,
                    min=0,
                    max=6,
                    optional=True,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
            ],
            outputs=[]
        )

    @classmethod
    def execute(cls,
                image: torch.Tensor,
                file_format: str,
                websocket_url: str,
                secret_token: str,
                request_job_id: str,
                oxipng_level: int = 0,
                **kwargs) -> None:
        try:
            epoch_time:int = int(time.time() * 1000)

            img_bytes = PixelSocketUtils.tensor_to_image_bytes(image, file_format, oxipng_level)
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

        return comfy_api_io.NodeOutput(image)
