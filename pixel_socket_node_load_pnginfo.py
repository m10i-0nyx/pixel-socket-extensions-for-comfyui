import json
import os
from typing import Any
import re

from PIL import Image
import piexif

from comfy_api.latest import io as comfy_api_io, ui as comfy_api_ui # pyright: ignore[reportMissingImports]
import folder_paths # pyright: ignore[reportMissingImports]

class PixelSocketLoadImageInfoNode(comfy_api_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_api_io.Schema:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])

        return comfy_api_io.Schema(
            node_id="PixelSocketLoadImageInfoNode",
            display_name="Load Image info Node",
            category="PixelSocket/Load",
            inputs=[
                comfy_api_io.Combo.Input("image_file",
                    upload=comfy_api_io.UploadType.image,
                    image_folder=comfy_api_io.FolderType.input,
                    options=sorted(files)
                )
            ],
            outputs=[
                comfy_api_io.String.Output("positive_prompt"),
                comfy_api_io.String.Output("negative_prompt"),
                comfy_api_io.String.Output("metadata"),
            ]
        )

    @classmethod
    def execute(cls, image_file, **kwargs) -> Any:
        def parse_geninfo(geninfo: str):
            """
            geninfo テキストをパースして positive_prompt, negative_prompt, メタデータを抽出
            <lora:...> 形式のテキストを除外
            """
            # "Negative prompt:" で分割
            if "Negative prompt:" in geninfo:
                parts = geninfo.split("Negative prompt:", 1)
                positive_prompt = parts[0].strip()

                negative_part = parts[1]

                # メタデータの開始位置を検出（最初のメタデータキーを探す）
                # メタデータキーは "キー名: 値" の形式で、通常キー名は複数単語
                metadata_match = re.search(r'\s\w+\s*:', negative_part)

                if metadata_match:
                    # メタデータの開始位置
                    metadata_start = metadata_match.start()
                    negative_prompt = negative_part[:metadata_start].strip()
                    metadata_text = negative_part[metadata_start:]
                else:
                    # メタデータが見つからない場合は全て negative_prompt
                    negative_prompt = negative_part.strip()
                    metadata_text = ""
            else:
                # "Negative prompt:" がない場合は全体が positive_prompt
                positive_prompt = geninfo.strip()
                negative_prompt = ""
                metadata_text = ""

            # <lora:...> 形式のテキストを除外
            positive_prompt = re.sub(r'<lora:[^>]+>\s*', '', positive_prompt)
            negative_prompt = re.sub(r'<lora:[^>]+>\s*', '', negative_prompt)

            # メタデータを抽出
            metadata = {}
            if metadata_text:
                # "key: value" 形式で分割
                # ただしクォートで囲まれた値や複雑な値に対応
                pairs = re.findall(r'(\w+(?:\s+\w+)*?)\s*:\s*([^,]+?)(?=,\s*\w+\s*:|$)', metadata_text)
                for key, value in pairs:
                    key = key.strip()
                    value = value.strip().strip('"')
                    metadata[key] = value

            return positive_prompt.strip(), negative_prompt.strip(), json.dumps(metadata)


        image_path = folder_paths.get_annotated_filepath(image_file)
        print(f"[PixelSocketLoadImageInfoNode] Loading image info from: {image_path}")

        image = Image.open(image_path)

        items = (image.info or {}).copy()
        positive_prompt: str = ""
        negative_prompt: str = ""
        metadata_text: str = ""

        if items.get("Software", None) == "NovelAI":
            comment_dict = json.loads(items.get("Comment", "{}"))

            if "v4_prompt" in comment_dict and "caption" in comment_dict["v4_prompt"]:
                positive_prompt = comment_dict.get("base_caption", "")
            elif "prompt" in comment_dict:
                positive_prompt = comment_dict.get("prompt", "")

            if "v4_negative_prompt" in comment_dict and "caption" in comment_dict["v4_prompt"]:
                negative_prompt = comment_dict.get("base_caption", "")
            elif "negative_prompt" in comment_dict:
                negative_prompt = comment_dict.get("negative_prompt", "")
            elif "uc" in comment_dict:
                negative_prompt = comment_dict.get("uc", "")

            del comment_dict["prompt"]
            del comment_dict["negative_prompt"]
            del comment_dict["uc"]
            del comment_dict["v4_prompt"]
            del comment_dict["v4_negative_prompt"]

            metadata_text = json.dumps(comment_dict, ensure_ascii=False)

        elif "exif" in items:
            geninfo: str = ""
            exif_data = items["exif"]
            try:
                exif = piexif.load(exif_data)
            except OSError:
                # memory / exif was not valid so piexif tried to read from a file
                exif = None
            exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
            try:
                exif_comment = piexif.helper.UserComment.load(exif_comment) # type: ignore
            except ValueError:
                exif_comment = exif_comment.decode('utf8', errors="ignore")

            if exif_comment:
                geninfo = exif_comment
            # geninfo をパース
            (positive_prompt, negative_prompt, metadata_text) = parse_geninfo(geninfo or "")

        elif "comment" in items: # for gif
            geninfo: str = ""
            if isinstance(items["comment"], bytes):
                geninfo = items["comment"].decode('utf8', errors="ignore")
            else:
                geninfo = items["comment"]
            # geninfo をパース
            (positive_prompt, negative_prompt, metadata_text) = parse_geninfo(geninfo or "")


        # デバッグ/確認用に出力
        print(f"[PixelSocketLoadImageInfoNode] Positive Prompt: {positive_prompt}")
        print(f"[PixelSocketLoadImageInfoNode] Negative Prompt: {negative_prompt}")
        print(f"[PixelSocketLoadImageInfoNode] Metadata: {metadata_text}")

        return comfy_api_io.NodeOutput(positive_prompt, negative_prompt, metadata_text)
