from typing import Any
import base64
import io
import json
import numpy as np
import oxipng
import piexif
import time
import websocket

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import comfy # pyright: ignore[reportMissingImports]
from comfy_api.latest import ComfyExtension, io as comfy_api_io # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]
import msgpack
import zstd

class PixelSocketDeliveryImageNode(comfy_api_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_api_io.Schema:
        return comfy_api_io.Schema(
            node_id="PixelSocketDeliveryImageNode",
            display_name="Delivery Image Node",
            category="PixelSocket",
            is_output_node=True,
            inputs=[
                comfy_api_io.Image.Input("image"),
                comfy_api_io.Combo.Input("file_format",
                    options=["WEBP", "PNG"],
                    default="WEBP"
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
                comfy_api_io.Model.Input("checkpoint_name"),
                comfy_api_io.String.Input("positive_prompt",
                    default="",
                    multiline=True,
                    optional=False
                ),
                comfy_api_io.String.Input("negative_prompt",
                    default="",
                    multiline=True,
                    optional=False
                ),
                comfy_api_io.Int.Input("seed_value",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    step=1,
                    optional=False,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
                comfy_api_io.Int.Input("width",
                    default=512,
                    min=1,
                    max=8192,
                    step=8,
                    optional=False,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
                comfy_api_io.Int.Input("height",
                    default=512,
                    min=1,
                    max=8192,
                    step=8,
                    optional=False,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
                comfy_api_io.Int.Input("step",
                    default=20,
                    min=1,
                    max=100,
                    step=1,
                    optional=False,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
                comfy_api_io.Float.Input("cfg",
                    default=8.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    optional=False,
                    display_mode=comfy_api_io.NumberDisplay.number
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
                checkpoint_name: str,
                positive_prompt,
                negative_prompt,
                seed_value: int,
                width: int,
                height: int,
                step: int,
                cfg: float,
                oxipng_level: int = 0,
                **kwargs) -> None:
        try:
            epoch_time:int = int(time.time() * 1000)

            metadata: dict[str, Any] = {
                "checkpoint_name": checkpoint_name,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "seed_value": seed_value,
                "width": width,
                "height": height,
                "step": step,
                "cfg": cfg,
                "comfyui_version": getattr(comfy, "__version__", "unknown"),
            }

            img_bytes = PixelSocketExtensions.tensor_to_image_bytes(image, file_format, oxipng_level, metadata)
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
                    "promptParams": metadata
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

class PixelSocketLoadImageFromBase64Node(comfy_api_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_api_io.Schema:
        return comfy_api_io.Schema(
            node_id="PixelSocketLoadImageFromBase64Node",
            display_name="Load Image From Base64 Node",
            category="PixelSocket",
            is_output_node=True,
            inputs=[
                comfy_api_io.String.Input("image_base64",
                    default="<IMAGE_BASE64>",
                    multiline=True,
                    optional=False
                ),
            ],
            outputs=[
                comfy_api_io.Image.Output("image"),
                comfy_api_io.Mask.Output("mask"),
                comfy_api_io.Int.Output("width"),
                comfy_api_io.Int.Output("height"),
            ]
        )

    @classmethod
    def execute(cls,
                image_base64: str,
                **kwargs) -> None:

        try:
            # Decode base64 string
            image_data = base64.b64decode(image_base64)

            # Load image using PIL
            img = Image.open(io.BytesIO(image_data)).convert("RGBA")

            # Convert to tensor
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

            img_mask: torch.Tensor
            # Extract alpha channel as mask
            if img.mode == "RGBA" and img_array.shape[2] == 4:
                img_mask = img_array[:, :, 3]  # Alpha channel
                img_mask = torch.from_numpy(img_mask).unsqueeze(0)  # Add batch dimension
            else:
                # Create a full mask if no alpha channel
                img_mask = torch.ones((1, img.height, img.width), dtype=torch.float32)

            width = img.width if img else None
            height = img.height if img else None

            return comfy_api_io.NodeOutput(img_tensor, img_mask, width, height)

        except Exception:
            import traceback
            traceback.print_exc()

        return comfy_api_io.NodeOutput(None, None, None, None)

class PixelSocketExtensions(ComfyExtension):
    async def get_node_list(self) -> list[type[comfy_api_io.ComfyNode]]:
        return [
                    PixelSocketDeliveryImageNode,
                    PixelSocketLoadImageFromBase64Node
               ]

    @classmethod
    def tensor_to_image_bytes(cls, image: torch.Tensor, file_format: str, oxipng_level: int, metadata: dict[str, Any]) -> bytes:
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
            pnginfo = PngInfo()
            for key, value in metadata.items():
                pnginfo.add_text(key, str(value))

            img.save(buf, format="PNG", pnginfo=pnginfo)

            # Optimize PNG using oxipng
            if oxipng_level > 0 and oxipng_level <= 6:
                buf.seek(0)
                buf = io.BytesIO(oxipng.optimize_from_memory(buf.getvalue(), level=oxipng_level))

        elif file_format.lower() == "webp":
            exif_bytes = piexif.dump({
                "Exif": {
                    piexif.ExifIFD.UserComment: b"ASCII\x00\x00\x00" + json.dumps(metadata, ensure_ascii=True).encode('utf-8')
                },
            })
            img.save(buf, format="WEBP", optimize=True, lossless=True, exif=exif_bytes)

        else:
            raise ValueError("Unsupported format")

        return buf.getvalue()

async def comfy_entrypoint() -> ComfyExtension:
    return PixelSocketExtensions()
