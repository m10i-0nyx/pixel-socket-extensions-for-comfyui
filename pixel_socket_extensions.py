from typing import Any
import base64
import io
import json
import numpy as np
import oxipng
import piexif

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from comfy_api.latest import ComfyExtension, io as comfy_api_io # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]
import httpx

from .pixel_socket_node_delivery import PixelSocketDeliveryImageNode

class PixelSocketLoadImageFromUrlNode(comfy_api_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_api_io.Schema:
        return comfy_api_io.Schema(
            node_id="PixelSocketLoadImageFromUrlNode",
            display_name="Load Image From URL Node",
            category="PixelSocket/Load",
            is_output_node=True,
            inputs=[
                comfy_api_io.String.Input("image_url",
                    default="",
                    multiline=False,
                    optional=False
                )
            ],
            outputs=[
                comfy_api_io.Image.Output("image"),
            ]
        )

    @classmethod
    def execute(cls, image_url: str, **kwargs) -> None:
        try:
            img_data: bytes = b""
            if image_url.startswith("data:image/"):
                _, encoded = image_url.split(",", 1)
                img_data = base64.b64decode(encoded)

            elif image_url.startswith("http://") or image_url.startswith("https://"):
                response = httpx.get(image_url)
                response.raise_for_status()
                img_data = response.content
            else:
                print(f"[PixelSocketLoadImageFromUrlNode] WARNING: Unsupported URL scheme.")

            # Validate image data
            if not img_data or not cls._validate_image_data(img_data):
                print(f"[PixelSocketLoadImageFromUrlNode] WARNING: Invalid image data. Returning blank 1024x1024 image.")
                return PixelSocketExtensions.create_fallback_image()

            img = Image.open(io.BytesIO(img_data)).convert("RGBA")

            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            return comfy_api_io.NodeOutput(img_tensor)

        except Exception as ex:
            print(f"[PixelSocketLoadImageFromUrlNode] ERROR: {ex}")
            import traceback
            traceback.print_exc()

        return PixelSocketExtensions.create_fallback_image()

    @classmethod
    def _validate_image_data(cls, img_data: bytes) -> bool:
        """画像データが適切であるか判定"""
        MAX_SIZE = 10 * 1024 * 1024  # 10MB

        # ファイルサイズチェック
        if len(img_data) > MAX_SIZE:
            print(f"[PixelSocketLoadImageFromUrlNode] WARNING: Image size {len(img_data)} bytes exceeds 10MB limit")
            return False

        # 画像フォーマットチェック（マジックナンバー）
        if len(img_data) < 4:
            print(f"[PixelSocketLoadImageFromUrlNode] WARNING: Image data too small: {len(img_data)} bytes")
            return False

        # 既知の画像フォーマットのマジックナンバーをチェック
        magic_numbers = [
            (b'\x89PNG', 'PNG'),       # PNG
            (b'\xff\xd8\xff', 'JPEG'), # JPEG
            (b'GIF8', 'GIF'),          # GIF
            (b'RIFF', 'WebP'),         # WebP (RIFF format)
        ]

        is_valid_format = False
        for magic, fmt in magic_numbers:
            if img_data.startswith(magic):
                print(f"[PixelSocketLoadImageFromUrlNode] Valid {fmt} image detected ({len(img_data)} bytes)")
                is_valid_format = True
                break

        if not is_valid_format:
            print(f"[PixelSocketLoadImageFromUrlNode] WARNING: Unknown or unsupported image format")
            return False

        # PIL で開けるかテスト
        try:
            Image.open(io.BytesIO(img_data)).verify()
            return True
        except Exception as e:
            print(f"[PixelSocketLoadImageFromUrlNode] WARNING: Image verification failed: {e}")
            return False

class PixelSocketResizeImageNode(comfy_api_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_api_io.Schema:
        return comfy_api_io.Schema(
            node_id="PixelSocketResizeImageNode",
            display_name="Resize Image Node",
            category="PixelSocket/Processing",
            is_output_node=True,
            inputs=[
                comfy_api_io.Image.Input("image"),
                comfy_api_io.Int.Input("width",
                    default=1024,
                    min=0,
                    step=8,
                    optional=False,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
                comfy_api_io.Int.Input("height",
                    default=1024,
                    min=0,
                    step=8,
                    optional=False,
                    display_mode=comfy_api_io.NumberDisplay.number
                ),
            ],
            outputs=[
                comfy_api_io.Image.Output("image"),
                comfy_api_io.Int.Output("width"),
                comfy_api_io.Int.Output("height"),
            ]
        )

    @classmethod
    def execute(cls, image: torch.Tensor, width: int, height: int, **kwargs) -> None:
        try:
            img = PixelSocketExtensions.tensor_to_image(image)
            img = img.convert("RGBA")

            # アスペクト比を維持しながらwidth/height以内の最大サイズにリサイズ
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height

            if width / height > aspect_ratio:
                # 高さに合わせる
                new_height = height
                new_width = int(height * aspect_ratio)
            else:
                # 幅に合わせる
                new_width = width
                new_height = int(width / aspect_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            width_output, height_output = img.size

            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            return comfy_api_io.NodeOutput(img_tensor, width_output, height_output)

        except Exception as ex:
            print(f"[PixelSocketResizeImageNode] ERROR: {ex}")
            import traceback
            traceback.print_exc()

        return PixelSocketExtensions.create_fallback_image(width, height)

class PixelSocketExtensions(ComfyExtension):
    async def get_node_list(self) -> list[type[comfy_api_io.ComfyNode]]:
        return [
                    PixelSocketDeliveryImageNode,
                    PixelSocketLoadImageFromUrlNode,
                    PixelSocketResizeImageNode,
               ]

    @classmethod
    def tensor_to_image(cls, image: torch.Tensor) -> Image.Image:
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
        return img

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

    @classmethod
    def create_fallback_image(cls, width: int = 1024, height: int = 1024) -> comfy_api_io.NodeOutput:
        """空白イメージを生成"""
        blank_img = Image.new("RGBA", (width, height), color=(255, 255, 255, 255))
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return comfy_api_io.NodeOutput(img_tensor, width, height)

async def comfy_entrypoint() -> ComfyExtension:
    return PixelSocketExtensions()
