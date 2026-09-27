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
        self.completion_kwargs = kwargs
        self.messages = kwargs["messages"]
        return {"choices": [{"message": {"content": "two images"}}],
                "usage": {"completion_tokens": 3}}

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


class FakeSpecConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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
        speculative = types.ModuleType("llama_cpp.llama_speculative")
        speculative.SpecConfig = FakeSpecConfig
        speculative.SpeculativeType = types.SimpleNamespace(DRAFT_MTP=3)

        cls.original_modules = {name: sys.modules.get(name) for name in
                                ("folder_paths", "llama_cpp", "llama_cpp.llama_chat_format",
                                 "llama_cpp.llama_speculative")}
        sys.modules.update({"folder_paths": folder_paths, "llama_cpp": llama_cpp,
                            "llama_cpp.llama_chat_format": chat_format,
                            "llama_cpp.llama_speculative": speculative})
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
        self.assertTrue(options["seed"][1]["control_after_generate"])

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
        self.assertEqual(FakeLlama.instances[0].kwargs["flash_attn_type"], -1)
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

    def test_preset_and_seed_are_forwarded(self):
        node = self.nodes.QwenGGUFInference()
        node.infer(str(Path("Qwen-VL") / "Qwen3.5-Q4.gguf"), "None", "", "Focus on the sign.",
                   10, 0, 8192, 0, preset_prompt="Extract Text (OCR)", seed=12345)
        request = FakeLlama.instances[0].completion_kwargs
        self.assertEqual(request["seed"], 12345)
        self.assertIn("Transcribe all readable text", request["messages"][0]["content"])
        self.assertIn("Focus on the sign.", request["messages"][0]["content"])

    def test_attention_change_reloads_and_keep_off_releases(self):
        node = self.nodes.QwenGGUFInference()
        args = (str(Path("Qwen-VL") / "Qwen3.5-Q4.gguf"), "None", "", "hello", 10, 0, 8192, 0)
        node.infer(*args)
        first = FakeLlama.instances[0]
        node.infer(*args, attention_mode="disabled", keep_model_loaded=False)
        self.assertTrue(first.closed)
        self.assertEqual(FakeLlama.instances[1].kwargs["flash_attn_type"], 0)
        self.assertTrue(FakeLlama.instances[1].closed)
        self.assertIsNone(self.nodes._MODEL)

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

    def advanced_options(self, mmproj="None"):
        required = self.nodes.QwenGGUFInferenceAdvanced.INPUT_TYPES()["required"]
        options = {name: spec[1].get("default") if len(spec) > 1 and isinstance(spec[1], dict) else spec[0][0]
                   for name, spec in required.items()}
        options.update(model=str(Path("Qwen-VL") / "Qwen3.5-Q4.gguf"),
                       mmproj=mmproj, user_prompt="describe")
        return options

    def test_advanced_sampler_and_batch_limits(self):
        options = self.advanced_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        options.update(top_p=0.8, top_k=30, min_p=0.1, repetition_penalty=1.2,
                       frequency_penalty=0.4, presence_penalty=0.3,
                       mirostat_mode="v2", n_batch=1024, n_ubatch=256,
                       n_threads=4, image_max_tokens=600, max_images=2,
                       enable_thinking=True, reasoning_budget=80,
                       image=FakeTensor(np.zeros((4, 2, 2, 3), dtype=np.float32)))
        answer, reasoning, stats = self.nodes.QwenGGUFInferenceAdvanced().infer(**options)
        self.assertEqual((answer, reasoning), ("two images", ""))
        self.assertIn('"completion_tokens": 3', stats)
        instance = FakeLlama.instances[0]
        self.assertEqual(instance.kwargs["n_ubatch"], 256)
        self.assertEqual(instance.kwargs["n_threads"], 4)
        self.assertEqual(FakeHandler.instances[0].kwargs["image_max_tokens"], 600)
        self.assertTrue(FakeHandler.instances[0].kwargs["enable_thinking"])
        request = instance.completion_kwargs
        for name, value in (("top_p", 0.8), ("top_k", 30), ("min_p", 0.1),
                            ("repeat_penalty", 1.2), ("frequency_penalty", 0.4),
                            ("presence_penalty", 0.3), ("mirostat_mode", 2),
                            ("reasoning_budget", 80)):
            self.assertEqual(request[name], value)
        self.assertEqual([part["text"] for part in request["messages"][-1]["content"]
                          if part["type"] == "text"][1:], ["Image 1:", "Image 4:"])

    def test_mtp_uses_speculative_config_and_reloads(self):
        options = self.advanced_options()
        node = self.nodes.QwenGGUFInferenceAdvanced()
        node.infer(**options)
        first = FakeLlama.instances[0]
        options.update(mtp_draft_tokens=4, mtp_draft_p_min=0.15)
        node.infer(**options)
        self.assertTrue(first.closed)
        spec = FakeLlama.instances[1].kwargs["speculative"]
        self.assertEqual((spec.spec_type, spec.draft_n_max, spec.draft_p_min), (3, 4, 0.15))

    def test_thinking_requires_projector(self):
        options = self.advanced_options()
        options["enable_thinking"] = True
        with self.assertRaisesRegex(ValueError, "requires mmproj"):
            self.nodes.QwenGGUFInferenceAdvanced().infer(**options)


if __name__ == "__main__":
    unittest.main()
