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
    calls = []
    response = {"choices": [{"message": {"content": "two images"}}],
                "usage": {"completion_tokens": 3}}

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = None
        self.closed = False
        self.instances.append(self)

    def create_chat_completion(self, **kwargs):
        self.completion_kwargs = kwargs
        self.messages = kwargs["messages"]
        self.calls.append(kwargs)
        return self.response

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
        FakeLlama.calls.clear()
        FakeHandler.instances.clear()
        FakeLlama.response = {"choices": [{"message": {"content": "two images"}}],
                              "usage": {"completion_tokens": 3}}

    def test_inputs_and_preset_order(self):
        options = self.nodes.QwenGGUFInference.INPUT_TYPES()["required"]
        self.assertIn(str(Path("Qwen-VL") / "Qwen3.5-Q4.gguf"), options["model"][0])
        self.assertIn(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"), options["mmproj"][0])
        self.assertLess(list(options).index("user_prompt"), list(options).index("system_prompt"))
        self.assertEqual(options["attention_mode"][0], ["auto", "on", "off"])
        self.assertIn("Prompt Style - Cinematic", options["preset_prompt"][0])
        self.assertIn("parameters", self.nodes.QwenGGUFInference.INPUT_TYPES()["optional"])
        self.assertIn("video", self.nodes.QwenGGUFInference.INPUT_TYPES()["optional"])
        self.assertTrue(options["seed"][1]["control_after_generate"])
        param_names = list(self.nodes.QwenGGUFParameters.INPUT_TYPES()["required"])
        self.assertEqual(param_names[:2], ["enable_thinking", "reasoning_budget"])

    def base_options(self, mmproj="None"):
        return dict(model=str(Path("Qwen-VL") / "Qwen3.5-Q4.gguf"), mmproj=mmproj,
                    user_prompt="describe", system_prompt="system")

    def parameter_options(self):
        required = self.nodes.QwenGGUFParameters.INPUT_TYPES()["required"]
        return {name: spec[1]["default"] if len(spec) > 1 else spec[0][0]
                for name, spec in required.items()}

    def test_batch_is_one_ordered_multimodal_request(self):
        pixels = np.zeros((2, 2, 2, 3), dtype=np.float32)
        pixels[0, :, :, 0] = 1
        pixels[1, :, :, 1] = 1
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        options.update(preset_prompt="Compare Images", image=FakeTensor(pixels))
        result = self.nodes.QwenGGUFInference().infer(**options)

        self.assertEqual(result[:2], ("two images", ""))
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
        self.assertIn("Compare the images", content[0]["text"])
        for part, channel in ((content[2], 0), (content[4], 1)):
            url = part["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/png;base64,"))
            image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
            self.assertEqual(image.getpixel((0, 0))[channel], 255)

    def test_model_is_reused(self):
        node = self.nodes.QwenGGUFInference()
        for _ in range(2):
            node.infer(**self.base_options())
        self.assertEqual(len(FakeLlama.instances), 1)

    def test_preset_and_seed_are_forwarded(self):
        options = self.base_options()
        options.update(user_prompt="Focus on the sign.", preset_prompt="Extract Text (OCR)", seed=12345)
        self.nodes.QwenGGUFInference().infer(**options)
        request = FakeLlama.instances[0].completion_kwargs
        self.assertEqual(request["seed"], 12345)
        self.assertEqual(request["max_tokens"], 1024)
        self.assertIn("Transcribe all readable text", request["messages"][1]["content"])
        self.assertIn("Focus on the sign.", request["messages"][1]["content"])

    def test_attention_change_reloads_and_keep_off_releases(self):
        node = self.nodes.QwenGGUFInference()
        options = self.base_options()
        node.infer(**options)
        first = FakeLlama.instances[0]
        node.infer(**options, attention_mode="off", keep_model_loaded=False)
        self.assertTrue(first.closed)
        self.assertEqual(FakeLlama.instances[1].kwargs["flash_attn_type"], 0)
        self.assertTrue(FakeLlama.instances[1].closed)
        self.assertIsNone(self.nodes._MODEL)

    def test_image_requires_projector(self):
        image = FakeTensor(np.zeros((1, 1, 1, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "mmproj"):
            self.nodes.QwenGGUFInference().infer(**self.base_options(), image=image)

    def test_invalid_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid GGUF"):
            self.nodes._model_path("../outside.gguf")

    def test_connected_parameters_are_forwarded(self):
        params = self.parameter_options()
        params.update(top_p=0.8, top_k=50, min_p=0.1, typical_p=0.7,
                      repeat_penalty=1.2, frequency_penalty=0.4,
                      presence_penalty=0.3, mirostat_mode=2, n_batch=1024,
                      n_threads=4, image_max_tokens=600, max_frames=2,
                      enable_thinking=True, reasoning_budget=80)
        bundle = self.nodes.QwenGGUFParameters().build(**params)[0]
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        options.update(parameters=bundle,
                       image=FakeTensor(np.zeros((4, 2, 2, 3), dtype=np.float32)))
        answer, reasoning, stats = self.nodes.QwenGGUFInference().infer(**options)
        self.assertEqual((answer, reasoning), ("two images", ""))
        self.assertIn('"completion_tokens": 3', stats)
        instance = FakeLlama.instances[0]
        self.assertEqual(instance.kwargs["n_batch"], 1024)
        self.assertEqual(instance.kwargs["n_threads"], 4)
        self.assertEqual(FakeHandler.instances[0].kwargs["image_max_tokens"], 600)
        self.assertTrue(FakeHandler.instances[0].kwargs["enable_thinking"])
        request = instance.completion_kwargs
        for name, value in (("top_p", 0.8), ("top_k", 50), ("min_p", 0.1),
                            ("typical_p", 0.7),
                            ("repeat_penalty", 1.2), ("frequency_penalty", 0.4),
                            ("presence_penalty", 0.3), ("mirostat_mode", 2),
                            ("reasoning_budget", 80)):
            self.assertEqual(request[name], value)
        self.assertEqual([part["text"] for part in request["messages"][-1]["content"]
                          if part["type"] == "text"][1:], ["Image 1:", "Image 4:"])

    def test_mtp_uses_speculative_config_and_reloads(self):
        options = self.base_options()
        node = self.nodes.QwenGGUFInference()
        node.infer(**options)
        first = FakeLlama.instances[0]
        params = self.parameter_options()
        params.update(mtp_draft_tokens=4, mtp_draft_p_min=0.15)
        node.infer(**options, parameters=self.nodes.QwenGGUFParameters().build(**params)[0])
        self.assertTrue(first.closed)
        spec = FakeLlama.instances[1].kwargs["speculative"]
        self.assertEqual((spec.spec_type, spec.draft_n_max, spec.draft_p_min), (3, 4, 0.15))

    def test_thinking_requires_projector(self):
        options = self.base_options()
        params = self.parameter_options()
        params["enable_thinking"] = True
        with self.assertRaisesRegex(ValueError, "requires mmproj"):
            self.nodes.QwenGGUFInference().infer(**options, parameters=params)

    def test_video_frames_are_sampled_timestamped_and_resized(self):
        frames = FakeTensor(np.zeros((5, 4, 8, 3), dtype=np.float32))
        video = types.SimpleNamespace(get_components=lambda: types.SimpleNamespace(
            images=frames, frame_rate=2))
        params = self.parameter_options()
        params.update(inference_mode="all at once", max_frames=3, max_size=2)
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        answer, _, stats = self.nodes.QwenGGUFInference().infer(
            **options, video=video, parameters=params)
        self.assertEqual(answer, "two images")
        self.assertIn('"selected_frames": 3', stats)
        content = FakeLlama.calls[0]["messages"][-1]["content"]
        self.assertEqual([item["text"] for item in content if item["type"] == "text"][1:],
                         ["Frame 1 (0.00s):", "Frame 3 (1.00s):", "Frame 5 (2.00s):"])
        urls = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
        image = Image.open(io.BytesIO(base64.b64decode(urls[0].split(",", 1)[1])))
        self.assertEqual(max(image.size), 2)

    def test_one_by_one_summarizes_frame_observations(self):
        params = self.parameter_options()
        params.update(inference_mode="one by one", max_frames=2)
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        options.update(image=FakeTensor(np.zeros((4, 2, 2, 3), dtype=np.float32)),
                       parameters=params)
        _, _, stats = self.nodes.QwenGGUFInference().infer(**options)
        self.assertEqual(len(FakeLlama.calls), 3)
        self.assertIn("Image 1: two images", FakeLlama.calls[-1]["messages"][-1]["content"])
        self.assertIn("Image 4: two images", FakeLlama.calls[-1]["messages"][-1]["content"])
        self.assertIn('"inference_mode": "one_by_one"', stats)
        self.assertIn('"model_calls": 3', stats)

    def test_auto_mode_uses_frame_count_and_context(self):
        params = self.parameter_options()
        self.assertEqual(self.nodes._mode(2, params, 8192), "all_at_once")
        self.assertEqual(self.nodes._mode(12, params, 8192), "one_by_one")
        self.assertEqual(self.nodes._mode(4, params, 512), "one_by_one")

    def test_image_and_video_are_exclusive(self):
        frames = FakeTensor(np.zeros((1, 2, 2, 3), dtype=np.float32))
        video = types.SimpleNamespace(get_components=lambda: types.SimpleNamespace(
            images=frames, frame_rate=24))
        with self.assertRaisesRegex(ValueError, "either IMAGE or VIDEO"):
            self.nodes.QwenGGUFInference().infer(**self.base_options(), image=frames, video=video)


if __name__ == "__main__":
    unittest.main()
