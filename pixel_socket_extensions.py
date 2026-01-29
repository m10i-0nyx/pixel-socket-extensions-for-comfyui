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
                    default="",
                    multiline=True,
                    optional=False
                ),
            ],
            outputs=[
                comfy_api_io.Image.Output("image"),
                comfy_api_io.Int.Output("width"),
                comfy_api_io.Int.Output("height"),
            ]
        )

    @staticmethod
    def _decode_base64(image_base64: str) -> bytes:
        """Base64データをデコード"""
        if not image_base64 or not isinstance(image_base64, str):
            raise ValueError(f"Invalid base64 input: {type(image_base64)}")

        # ホワイトスペースを削除してからデコード
        image_base64 = image_base64.strip()

        # パディングを自動的に追加（必要な場合）
        missing_padding = len(image_base64) % 4
        if missing_padding:
            image_base64 += '=' * (4 - missing_padding)

        try:
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 format: {e}")

        if len(image_data) == 0:
            raise ValueError("Decoded base64 data is empty")
        return image_data

    @staticmethod
    def _load_image(image_data: bytes) -> Image.Image:
        """バイナリデータから画像をロード"""
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        return img

    @staticmethod
    def _normalize_dimensions(img: Image.Image) -> tuple[Image.Image, int, int]:
        """画像寸法をVAE互換にリサイズ"""
        MIN_DIMENSION = 64
        MULTIPLE_OF = 8

        original_size = img.size
        width, height = img.size
        print(f"[PixelSocketLoadImageFromBase64Node] Original: {width}x{height}")

        # 最小値チェック
        if width < 4 or height < 4:
            raise ValueError(f"Image too small: {width}x{height}. Minimum 4x4 required.")

        # 最小寸法を確保
        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            scale = max(MIN_DIMENSION / width, MIN_DIMENSION / height)
            width, height = int(width * scale), int(height * scale)
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            print(f"[PixelSocketLoadImageFromBase64Node] Upscaled to {width}x{height}")

        # 8の倍数に丸める
        width = ((width + 7) // MULTIPLE_OF) * MULTIPLE_OF
        height = ((height + 7) // MULTIPLE_OF) * MULTIPLE_OF
        width = max(width, MIN_DIMENSION)
        height = max(height, MIN_DIMENSION)

        # リサイズ
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            print(f"[PixelSocketLoadImageFromBase64Node] Resized to {width}x{height}")

        return img, width, height

    @staticmethod
    def _image_to_tensor(img: Image.Image) -> torch.Tensor:
        """画像をテンソルに変換"""
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)
        print(f"[PixelSocketLoadImageFromBase64Node] Tensor shape: {img_tensor.shape}")
        return img_tensor

    @staticmethod
    def _create_fallback_image() -> torch.Tensor:
        """フォールバック用の黒いテンソルを生成"""
        default_img = Image.new("RGB", (512, 512), color=(0, 0, 0))
        img_array = np.array(default_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)

    @classmethod
    def execute(cls, image_base64: str, **kwargs) -> None:
        try:
            image_data = cls._decode_base64(image_base64)
            img = cls._load_image(image_data)
            img, width, height = cls._normalize_dimensions(img)
            img_tensor = cls._image_to_tensor(img)
            return comfy_api_io.NodeOutput(img_tensor, width, height)

        except Exception as e:
            print(f"[PixelSocketLoadImageFromBase64Node] ERROR: {e}")
            import traceback
            traceback.print_exc()
            fallback_tensor = cls._create_fallback_image()
            return comfy_api_io.NodeOutput(fallback_tensor, 512, 512)

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
