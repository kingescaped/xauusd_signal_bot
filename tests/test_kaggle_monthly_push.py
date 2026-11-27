import ast
import io
import json
import os
import pathlib
import re
import unittest
from unittest import mock
import urllib.error

import kaggle_push


class _FakeHttpResponse:
    def __init__(self, body_obj):
        self._raw = json.dumps(body_obj).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class TestKaggleMonthlyPush(unittest.TestCase):
    def test_normalize_kaggle_token_strips_prefix_and_quotes(self):
        self.assertEqual(kaggle_push.normalize_kaggle_token("  KGAT_abc123  "), "KGAT_abc123")
        self.assertEqual(kaggle_push.normalize_kaggle_token("KAGGLE_API_TOKEN='KGAT_xyz'"), "KGAT_xyz")
        with self.assertRaises(ValueError):
            kaggle_push.normalize_kaggle_token("   ")

    def test_build_push_payload_enforces_private_t4_gpu_and_internet(self):
        create_payload = kaggle_push.build_push_payload("vinhquangvo", '{"cells":[]}', is_existing=False)
        self.assertTrue(create_payload["isPrivate"])
        self.assertTrue(create_payload["enableGpu"])
        self.assertTrue(create_payload["enableInternet"])
        self.assertEqual(create_payload["machineShape"], "NvidiaTeslaT4")
        self.assertNotIn("slug", create_payload)

        update_payload = kaggle_push.build_push_payload("vinhquangvo", '{"cells":[]}', is_existing=True)
        self.assertEqual(update_payload["slug"], f"vinhquangvo/{kaggle_push.DEFAULT_KERNEL_SLUG}")

    def test_push_monthly_finetune_kernel_creates_updates_and_retries_conflict(self):
        calls = []
        list_mode = {"state": "empty"}

        def fake_urlopen(req, timeout=30):
            body = json.loads(req.data.decode("utf-8")) if req.data else None
            calls.append((req.full_url, req.headers.get("Authorization"), body))
            if req.full_url.endswith("/hello"):
                return _FakeHttpResponse({"userName": "vinhquangvo"})
            if "/kernels/list" in req.full_url:
                if list_mode["state"] == "existing":
                    return _FakeHttpResponse([{"ref": f"vinhquangvo/{kaggle_push.DEFAULT_KERNEL_SLUG}"}])
                return _FakeHttpResponse([])
            if list_mode["state"] == "conflict_once" and "slug" not in (body or {}):
                return _FakeHttpResponse({"error": "Title already in use"})
            return _FakeHttpResponse({
                "ref": f"vinhquangvo/{kaggle_push.DEFAULT_KERNEL_SLUG}",
                "url": f"https://www.kaggle.com/code/vinhquangvo/{kaggle_push.DEFAULT_KERNEL_SLUG}",
                "versionNumber": 2 if (body and "slug" in body) else 1,
                "error": None,
            })

        with (
            mock.patch.dict(os.environ, {"KAGGLE_USERNAME": ""}, clear=False),
            mock.patch.object(kaggle_push.urllib.request, "urlopen", side_effect=fake_urlopen),
        ):
            res_create = kaggle_push.push_monthly_finetune_kernel(token="KAGGLE_API_TOKEN=KGAT_dummy_secret")
            list_mode["state"] = "existing"
            res_update = kaggle_push.push_monthly_finetune_kernel(token="KGAT_dummy_secret")
            list_mode["state"] = "conflict_once"
            res_retry = kaggle_push.push_monthly_finetune_kernel(token="KGAT_dummy_secret")

        self.assertEqual(res_create["versionNumber"], 1)
        self.assertEqual(res_update["versionNumber"], 2)
        self.assertEqual(res_retry["versionNumber"], 2)
        self.assertIn("group=profile", calls[1][0])
        self.assertEqual(calls[0][1], "Bearer KGAT_dummy_secret")
        self.assertNotIn("slug", calls[2][2])
        self.assertEqual(calls[5][2]["slug"], f"vinhquangvo/{kaggle_push.DEFAULT_KERNEL_SLUG}")
        self.assertEqual(calls[-1][2]["slug"], f"vinhquangvo/{kaggle_push.DEFAULT_KERNEL_SLUG}")

    def test_push_raises_on_kaggle_error_or_http_error_without_exposing_token(self):
        def reject_urlopen(req, timeout=30):
            if req.full_url.endswith("/hello"):
                return _FakeHttpResponse({"userName": "vinhquangvo"})
            if "/kernels/list" in req.full_url:
                return _FakeHttpResponse([{"ref": f"vinhquangvo/{kaggle_push.DEFAULT_KERNEL_SLUG}"}])
            return _FakeHttpResponse({"error": "GPU quota or phone verification needed"})

        with mock.patch.object(kaggle_push.urllib.request, "urlopen", side_effect=reject_urlopen):
            with self.assertRaises(RuntimeError) as ctx:
                kaggle_push.push_monthly_finetune_kernel(token="KGAT_super_secret")
        self.assertIn("phone verification", str(ctx.exception))
        self.assertNotIn("KGAT_super_secret", str(ctx.exception))

        def http_401(req, timeout=30):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad token"))

        with mock.patch.object(kaggle_push.urllib.request, "urlopen", side_effect=http_401):
            with self.assertRaises(RuntimeError) as ctx401:
                kaggle_push.push_monthly_finetune_kernel(token="KGAT_super_secret")
        self.assertNotIn("KGAT_super_secret", str(ctx401.exception))

    def test_notebook_syntax_workflow_config_and_zero_hardcoded_tokens(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        nb_path = root / "kaggle" / "finetune_kaggle_monthly.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 4)
        full_nb_src = ""
        for cell in code_cells:
            lines = [ln for ln in cell["source"] if not ln.lstrip().startswith(("!", "%"))]
            cell_src = "".join(lines)
            ast.parse(cell_src)
            full_nb_src += cell_src + "\n"
        self.assertIn("UserSecretsClient", full_nb_src)

        wf_path = root / ".github" / "workflows" / "monthly_finetune.yml"
        wf_text = wf_path.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", wf_text)
        self.assertIn("secrets.KAGGLE_API_TOKEN", wf_text)
        self.assertIn("monthly-finetune-", wf_text)

        token_pattern = re.compile(r"KGAT_[0-9a-fA-F]{20,}")
        for file_path in root.rglob("*"):
            if not file_path.is_file() or ".git" in file_path.parts or file_path.name == ".env":
                continue
            if file_path.suffix in (".py", ".yml", ".ipynb", ".md", ".json", ".bat"):
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                self.assertIsNone(token_pattern.search(content), f"Real Kaggle token leaked in {file_path}")


if __name__ == "__main__":
    unittest.main()
