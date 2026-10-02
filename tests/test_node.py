"""Contract checks that run without ComfyUI or large model weights."""

import base64
import importlib.util
import io
import json
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
    raw_calls = []
    prompt_length = 24
    resets = 0
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
        return self.create_completion(prompt=[0] * self.prompt_length,
                                      **{k: v for k, v in kwargs.items() if k != "messages"})

    def create_completion(self, **kwargs):
        self.raw_calls.append(kwargs)
        return self.response

    def n_ctx(self):
        return self.kwargs["n_ctx"]

    def reset(self):
        self.resets += 1

    def detokenize(self, tokens, **kwargs):
        handler = self.kwargs.get("chat_handler")
        thinking = (handler.kwargs["enable_thinking"] if handler else
                    self.kwargs.get("chat_handler_kwargs", {}).get("extra_template_arguments", {}).get("enable_thinking"))
        return b"assistant\n<think>\n" if thinking else b"assistant\n<think>\n</think>\n"

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
        FakeLlama.raw_calls.clear()
        FakeLlama.prompt_length = 24
        FakeLlama.resets = 0
        FakeHandler.instances.clear()
        FakeLlama.response = {"choices": [{"message": {"content": "two images"}}],
                              "usage": {"completion_tokens": 3}}

    def test_response_excludes_thinking(self):
        cases = [
            ("<think>private reasoning</think>Final answer", "Final answer", "private reasoning"),
            ("private reasoning</think>\nFinal answer", "Final answer", "private reasoning"),
            ("<think>first</think><think>second</think>Final", "Final", "first\n\nsecond"),
            ("<think>unfinished reasoning", "", "unfinished reasoning"),
            ("<THINK>reasoning</THINK>Final", "Final", "reasoning"),
            ("Final answer", "Final answer", ""),
        ]
        for content, answer, reasoning in cases:
            with self.subTest(content=content):
                result = {"choices": [{"message": {"content": content}}]}
                self.assertEqual(self.nodes._response_parts(result), (answer, reasoning))

    def test_response_keeps_separate_reasoning(self):
        result = {"choices": [{"message": {"content": "Final", "reasoning_content": "Private"}}]}
        self.assertEqual(self.nodes._response_parts(result), ("Final", "Private"))

    def test_thinking_truncation_reports_missing_final_answer(self):
        result = {"choices": [{"message": {"content": "<think>unfinished"}, "finish_reason": "length"}]}
        with self.assertRaisesRegex(RuntimeError, "produced no final answer"):
            self.nodes._response_parts(result)

    def test_inputs_and_preset_order(self):
        options = self.nodes.QwenGGUFInference.INPUT_TYPES()["required"]
        self.assertIn(str(Path("Qwen-VL") / "Qwen3.5-Q4.gguf"), options["model"][0])
        self.assertEqual(len(self.nodes.DOWNLOAD_PRESETS), 4)
        self.assertTrue(all(preset in options["model"][0]
                            for preset in self.nodes.DOWNLOAD_PRESETS))
        self.assertIn(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"), options["mmproj"][0])
        self.assertLess(list(options).index("user_prompt"), list(options).index("system_prompt"))
        self.assertEqual(options["attention_mode"][0], ["auto", "on", "off"])
        self.assertIn("Prompt Style - Cinematic", options["preset_prompt"][0])
        self.assertIn("parameters", self.nodes.QwenGGUFInference.INPUT_TYPES()["optional"])
        self.assertIn("video", self.nodes.QwenGGUFInference.INPUT_TYPES()["optional"])
        self.assertEqual(self.nodes.QwenGGUFInference.INPUT_TYPES()["optional"]["video"], ("VIDEO",))
        self.assertTrue(options["seed"][1]["control_after_generate"])
        self.assertFalse(options["enable_thinking"][1]["default"])
        self.assertEqual(options["thinking_level"][0], ["auto", "low", "medium", "high", "custom"])
        param_names = list(self.nodes.QwenGGUFParameters.INPUT_TYPES()["required"])
        self.assertEqual(param_names[:2], ["enable_thinking", "reasoning_budget"])
        self.assertEqual(self.nodes.QwenGGUFParameters.INPUT_TYPES()["required"]["inference_mode"][0],
                         ["auto", "one by one", "images", "video"])

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

    def test_dynamic_gpu_backend_is_registered_before_offload_check(self):
        llama_cpp = sys.modules["llama_cpp"]
        original_support = llama_cpp.llama_supports_gpu_offload
        original_init = getattr(llama_cpp, "llama_backend_init", None)
        original_file = getattr(llama_cpp, "__file__", None)
        original_ggml = sys.modules.get("llama_cpp._ggml")
        state = {"loaded": False, "initialized": False}
        try:
            package = Path(self.temp_dir.name) / "probe_package"
            (package / "lib").mkdir(parents=True, exist_ok=True)
            llama_cpp.__file__ = str(package / "__init__.py")
            llama_cpp.llama_supports_gpu_offload = lambda: state["loaded"]
            llama_cpp.llama_backend_init = lambda: state.update(initialized=True)
            ggml = types.ModuleType("llama_cpp._ggml")

            def load_backend(path):
                self.assertTrue(state["initialized"])
                self.assertIn(b"lib", path.value)
                state["loaded"] = True

            ggml.ggml_backend_load_all_from_path = load_backend
            sys.modules["llama_cpp._ggml"] = ggml
            self.assertTrue(self.nodes._gpu_offload_available(llama_cpp))
        finally:
            llama_cpp.llama_supports_gpu_offload = original_support
            if original_init is None:
                if hasattr(llama_cpp, "llama_backend_init"):
                    del llama_cpp.llama_backend_init
            else:
                llama_cpp.llama_backend_init = original_init
            if original_file is None:
                del llama_cpp.__file__
            else:
                llama_cpp.__file__ = original_file
            if original_ggml is None:
                sys.modules.pop("llama_cpp._ggml", None)
            else:
                sys.modules["llama_cpp._ggml"] = original_ggml

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

    def test_download_preset_reuses_complete_model_and_projector(self):
        preset, filename = next(iter(self.nodes.DOWNLOAD_PRESETS.items()))
        projector = self.nodes.HF_MMPROJ
        previous_dir = self.nodes.HF_MODEL_DIR
        previous_sizes = dict(self.nodes.HF_FILE_SIZES)
        previous_hub = sys.modules.get("huggingface_hub")
        calls = []
        try:
            with tempfile.TemporaryDirectory() as directory:
                self.nodes.HF_MODEL_DIR = Path(directory)
                self.nodes.HF_FILE_SIZES[filename] = 8
                self.nodes.HF_FILE_SIZES[projector] = 8
                hub = types.ModuleType("huggingface_hub")

                def download(**kwargs):
                    calls.append(kwargs)
                    path = Path(kwargs["local_dir"]) / kwargs["filename"]
                    path.write_bytes(b"GGUFtest")
                    return str(path)

                hub.hf_hub_download = download
                sys.modules["huggingface_hub"] = hub
                model_path, mmproj_path = self.nodes._download_preset(preset)
                self.assertEqual(Path(model_path).name, filename)
                self.assertEqual(Path(mmproj_path).name, projector)
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(call["repo_id"] == self.nodes.HF_REPO_ID and
                                    call["revision"] == self.nodes.HF_REVISION for call in calls))
                self.nodes._download_preset(preset)
                self.assertEqual(len(calls), 2)
        finally:
            self.nodes.HF_MODEL_DIR = previous_dir
            self.nodes.HF_FILE_SIZES.clear()
            self.nodes.HF_FILE_SIZES.update(previous_sizes)
            if previous_hub is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = previous_hub

    def test_download_preset_uses_generic_qwen38_handler(self):
        original = self.nodes._download_preset
        captured = []
        try:
            def fake_download(choice):
                captured.append(choice)
                return ("Qwen3.8-27B-UD-IQ2_S.gguf", "mmproj-F16.gguf")

            self.nodes._download_preset = fake_download
            preset = next(iter(self.nodes.DOWNLOAD_PRESETS))
            image = FakeTensor(np.zeros((1, 2, 2, 3), dtype=np.float32))
            params = self.parameter_options()
            params["enable_thinking"] = True
            self.nodes.QwenGGUFInference().infer(
                model=preset, mmproj="None", user_prompt="describe", system_prompt="system",
                image=image, parameters=params,
            )
            self.assertEqual(captured, [preset])
            kwargs = FakeLlama.instances[0].kwargs
            self.assertEqual(kwargs["mmproj_path"], "mmproj-F16.gguf")
            self.assertIsNone(kwargs["chat_handler"])
            self.assertTrue(kwargs["chat_handler_kwargs"]["extra_template_arguments"]["enable_thinking"])
            self.assertEqual(len(FakeHandler.instances), 0)
        finally:
            self.nodes._download_preset = original

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

    def test_main_thinking_switch_overrides_parameters_and_reloads(self):
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        params = self.parameter_options()
        params.update(enable_thinking=True, reasoning_budget=80)
        node = self.nodes.QwenGGUFInference()
        node.infer(**options, parameters=params, enable_thinking=False)
        self.assertFalse(FakeHandler.instances[-1].kwargs["enable_thinking"])
        self.assertEqual(FakeLlama.calls[-1]["reasoning_budget"], 0)
        first = FakeLlama.instances[-1]
        params["enable_thinking"] = False
        node.infer(**options, parameters=params, enable_thinking=True)
        self.assertTrue(first.closed)
        self.assertTrue(FakeHandler.instances[-1].kwargs["enable_thinking"])
        self.assertEqual(FakeLlama.calls[-1]["reasoning_budget"], 80)

    def test_thinking_levels_reserve_final_answer_and_reuse_model(self):
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        node = self.nodes.QwenGGUFInference()
        for level, budget in (("auto", 512), ("low", 256), ("medium", 1024), ("high", 4096)):
            with self.subTest(level=level):
                _, _, stats = node.infer(**options, enable_thinking=True, thinking_level=level)
                call = FakeLlama.raw_calls[-1]
                self.assertEqual(call["reasoning_budget"], budget)
                self.assertEqual(call["max_tokens"], 1024 + budget + 64)
                self.assertTrue(call["reasoning_start_in_prompt"])
                self.assertIn("final answer", call["reasoning_budget_message"])
                self.assertEqual(json.loads(stats)["thinking"]["calls"][-1]["reasoning_budget"], budget)
        self.assertEqual(len(FakeLlama.instances), 1)
        self.assertNotIn("create_completion", vars(FakeLlama.instances[0]))

    def test_thinking_budget_shrinks_for_actual_prompt_size(self):
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        FakeLlama.prompt_length = 6800
        _, _, stats = self.nodes.QwenGGUFInference().infer(
            **options, enable_thinking=True, thinking_level="high")
        call = FakeLlama.raw_calls[-1]
        self.assertEqual(call["reasoning_budget"], 304)
        self.assertEqual(call["max_tokens"], 1392)
        self.assertEqual(json.loads(stats)["thinking"]["calls"][-1]["prompt_tokens"], 6800)

    def test_per_frame_thinking_is_bounded_separately(self):
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        params = self.parameter_options()
        params["inference_mode"] = "one by one"
        self.nodes.QwenGGUFInference().infer(
            **options, image=FakeTensor(np.zeros((2, 2, 2, 3), dtype=np.float32)),
            parameters=params, enable_thinking=True, thinking_level="high")
        self.assertEqual([call["reasoning_budget"] for call in FakeLlama.raw_calls], [64, 64, 4096])
        self.assertEqual([call["max_tokens"] for call in FakeLlama.raw_calls], [384, 384, 5184])

    def test_custom_zero_disables_thinking_and_ignores_budget_when_off(self):
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        params = self.parameter_options()
        params["reasoning_budget"] = 0
        node = self.nodes.QwenGGUFInference()
        node.infer(**options, parameters=params, enable_thinking=True, thinking_level="custom")
        self.assertFalse(FakeHandler.instances[-1].kwargs["enable_thinking"])
        node.infer(**options, enable_thinking=False, thinking_level="high")
        self.assertEqual(FakeLlama.raw_calls[-1]["reasoning_budget"], 0)
        self.assertEqual(FakeLlama.raw_calls[-1]["max_tokens"], 1024)

    def test_context_overflow_restores_completion_adapter(self):
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        FakeLlama.prompt_length = 8192
        with self.assertRaisesRegex(ValueError, "Input fills the context"):
            self.nodes.QwenGGUFInference().infer(
                **options, enable_thinking=True, thinking_level="high")
        self.assertNotIn("create_completion", vars(FakeLlama.instances[-1]))

    def test_prefilled_thinking_truncation_does_not_leak_to_response(self):
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        FakeLlama.response = {"choices": [{"message": {"content": "unfinished thoughts"}, "finish_reason": "length"}]}
        with self.assertRaisesRegex(RuntimeError, "produced no final answer"):
            self.nodes.QwenGGUFInference().infer(
                **options, enable_thinking=True, thinking_level="custom")
        self.assertNotIn("create_completion", vars(FakeLlama.instances[-1]))
        self.assertEqual(FakeLlama.instances[-1].resets, 1)

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
        params.update(inference_mode="video", max_frames=3, max_size=2, video_fps=10)
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        answer, _, stats = self.nodes.QwenGGUFInference().infer(
            **options, video=video, parameters=params)
        self.assertEqual(answer, "two images")
        self.assertIn('"selected_frames": 3', stats)
        self.assertIn('"total_frames": 5', stats)
        self.assertIn('"inference_mode": "video"', stats)
        self.assertIn("Treat the ordered frames as one video sequence",
                      FakeLlama.calls[0]["messages"][0]["content"])
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
        self.assertEqual(self.nodes._mode(2, params, 8192), "images")
        self.assertEqual(self.nodes._mode(2, params, 8192, has_video=True), "video")
        self.assertEqual(self.nodes._mode(12, params, 8192), "one_by_one")
        self.assertEqual(self.nodes._mode(4, params, 512), "one_by_one")

    def test_video_over_limit_samples_across_entire_clip(self):
        frames = FakeTensor(np.zeros((240, 2, 2, 3), dtype=np.float32))
        video = types.SimpleNamespace(get_components=lambda: types.SimpleNamespace(
            images=frames, frame_rate=24))
        params = self.parameter_options()
        params.update(inference_mode="video", max_frames=24)
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        _, _, stats = self.nodes.QwenGGUFInference().infer(
            **options, video=video, parameters=params)
        labels = [item["text"] for item in FakeLlama.calls[0]["messages"][-1]["content"]
                  if item["type"] == "text"][1:]
        self.assertEqual(len(labels), 24)
        self.assertEqual(labels[0], "Frame 1 (0.00s):")
        self.assertEqual(labels[-1], "Frame 240 (9.96s):")
        self.assertIn('"total_frames": 240', stats)
        self.assertIn('"selected_frames": 24', stats)

    def test_image_and_video_are_exclusive(self):
        frames = FakeTensor(np.zeros((1, 2, 2, 3), dtype=np.float32))
        video = types.SimpleNamespace(get_components=lambda: types.SimpleNamespace(
            images=frames, frame_rate=24))
        with self.assertRaisesRegex(ValueError, "either image or video"):
            self.nodes.QwenGGUFInference().infer(**self.base_options(), image=frames, video=video)

    def test_video_image_batch_without_fps_uses_frame_order(self):
        frames = FakeTensor(np.zeros((2, 2, 2, 3), dtype=np.float32))
        params = self.parameter_options()
        params["inference_mode"] = "video"
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        self.nodes.QwenGGUFInference().infer(**options, image=frames, parameters=params)
        content = FakeLlama.calls[0]["messages"][-1]["content"]
        self.assertEqual([item["text"] for item in content if item["type"] == "text"][1:],
                         ["Frame 1:", "Frame 2:"])

    def test_image_video_frames_with_fps_use_video_auto_mode(self):
        frames = FakeTensor(np.zeros((2, 2, 2, 3), dtype=np.float32))
        params = self.parameter_options()
        params["video_fps"] = 2
        options = self.base_options(str(Path("Qwen-VL") / "mmproj-Qwen3.5.gguf"))
        _, _, stats = self.nodes.QwenGGUFInference().infer(
            **options, image=frames, parameters=params)
        self.assertIn('"inference_mode": "video"', stats)
        content = FakeLlama.calls[0]["messages"][-1]["content"]
        self.assertEqual([item["text"] for item in content if item["type"] == "text"][1:],
                         ["Frame 1 (0.00s):", "Frame 2 (0.50s):"])


if __name__ == "__main__":
    unittest.main()
