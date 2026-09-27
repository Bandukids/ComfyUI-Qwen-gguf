"""Contract checks that run without ComfyUI or large model weights."""

import base64
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
from PIL import Image


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    def __iter__(self):
        return (FakeTensor(frame) for frame in self.values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def clamp(self, minimum, maximum):
        return FakeTensor(np.clip(self.values, minimum, maximum))

    def __mul__(self, value):
        return FakeTensor(self.values * value)

    def byte(self):
        return FakeTensor(self.values.astype(np.uint8))

    def numpy(self):
        return self.values


class FakeLlama:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = None
        self.closed = False
        self.instances.append(self)

    def create_chat_completion(self, **kwargs):
        self.messages = kwargs["messages"]
        return {"choices": [{"message": {"content": "two images"}}]}

    def close(self):
        self.closed = True


class FakeHandler:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.instances.append(self)

    def close(self):
        self.closed = True


class QwenNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        model_dir = root / "LLM" / "Qwen-VL"
        model_dir.mkdir(parents=True)
        (model_dir / "Qwen3.5-Q4.gguf").touch()
        (model_dir / "mmproj-Qwen3.5.gguf").touch()

        folder_paths = types.ModuleType("folder_paths")
        folder_paths.models_dir = str(root)
        folder_paths.add_model_folder_path = lambda *_: None
        folder_paths.get_filename_list = lambda *_: [str(Path("Qwen-VL") / path.name) for path in model_dir.glob("*.gguf")]
        folder_paths.get_full_path = lambda _, name: str(root / "LLM" / name) if (root / "LLM" / name).is_file() else None
        llama_cpp = types.ModuleType("llama_cpp")
        llama_cpp.Llama = FakeLlama
        llama_cpp.llama_supports_gpu_offload = lambda: False
        chat_format = types.ModuleType("llama_cpp.llama_chat_format")
        chat_format.Qwen35ChatHandler = FakeHandler

        cls.original_modules = {name: sys.modules.get(name) for name in
                                ("folder_paths", "llama_cpp", "llama_cpp.llama_chat_format")}
        sys.modules.update({"folder_paths": folder_paths, "llama_cpp": llama_cpp,
                            "llama_cpp.llama_chat_format": chat_format})
        spec = importlib.util.spec_from_file_location("qwen_test_nodes", Path(__file__).parents[1] / "nodes.py")
        cls.nodes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.nodes)

    @classmethod
    def tearDownClass(cls):
        cls.nodes._close_model()
        for name, original in cls.original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        cls.temp_dir.cleanup()

    def setUp(self):
        self.nodes._close_model()
        FakeLlama.instances.clear()
        FakeHandler.instances.clear()

    def test_existing_qwenvl_model_is_listed(self):
        options = self.nodes.QwenGGUFInference.INPUT_TYPES()["required"]
        self.assertIn(str(Path("Qwen-VL") / "Qwen3.5-Q4.gguf"), options["model"][0])
        self.assertIn(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"), options["mmproj"][0])

    def test_batch_is_one_ordered_multimodal_request(self):
        pixels = np.zeros((2, 2, 2, 3), dtype=np.float32)
        pixels[0, :, :, 0] = 1
        pixels[1, :, :, 1] = 1
        result = self.nodes.QwenGGUFInference().infer(
            str(Path("Qwen-VL") / "Qwen3.5-Q4.gguf"),
            str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"),
            "system", "compare", 128, 0.5, 8192, 99, FakeTensor(pixels),
        )

        self.assertEqual(result, ("two images",))
        self.assertEqual(len(FakeLlama.instances), 1)
        self.assertEqual(FakeLlama.instances[0].kwargs["n_gpu_layers"], 0)
        self.assertEqual(FakeHandler.instances[0].kwargs["enable_thinking"], False)
        messages = FakeLlama.instances[0].messages
        self.assertEqual(messages[0], {"role": "system", "content": "system"})
        content = messages[1]["content"]
        self.assertEqual([part["type"] for part in content],
                         ["text", "text", "image_url", "text", "image_url"])
        self.assertEqual(content[1]["text"], "Image 1:")
        self.assertEqual(content[3]["text"], "Image 2:")
        for part, channel in ((content[2], 0), (content[4], 1)):
            url = part["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/png;base64,"))
            image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
            self.assertEqual(image.getpixel((0, 0))[channel], 255)

    def test_model_is_reused(self):
        node = self.nodes.QwenGGUFInference()
        for _ in range(2):
            node.infer(str(Path("Qwen-VL") / "Qwen3.5-Q4.gguf"), "None", "", "hello",
                       10, 0, 8192, 0)
        self.assertEqual(len(FakeLlama.instances), 1)

    def test_image_requires_projector(self):
        image = FakeTensor(np.zeros((1, 1, 1, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "mmproj"):
            self.nodes.QwenGGUFInference().infer(
                str(Path("Qwen-VL") / "Qwen3.5-Q4.gguf"), "None", "", "hello",
                10, 0, 8192, 0, image,
            )

    def test_invalid_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid GGUF"):
            self.nodes._model_path("../outside.gguf")


if __name__ == "__main__":
    unittest.main()
