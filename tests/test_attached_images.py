import json, os, subprocess, sys, tempfile, time, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.join(os.path.dirname(HERE), "hook")
ID = "local_11111111-2222-3333-4444-555555555555"


def _write_text(path, text):
    with open(path, "w") as f:
        f.write(text)


def _write_json(path, obj):
    _write_text(path, json.dumps(obj))


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _stop(srv):
    srv.shutdown()
    srv.server_close()


class AttachedImageTest(unittest.TestCase):
    """Images the autoswitch-inline plugin passes as `images` (a message that is only screenshots): their OCR text is
    scored, an image with no readable text and an empty text prompt never get a target. OCR is a fake ACPMH."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.environ["AUTOSWITCH_DIR"] = self.d
        os.environ["AUTOSWITCH_SESSIONS"] = os.path.join(self.d, "s")
        os.environ["CLAUDE_CODE_ENTRYPOINT"] = "claude-desktop"
        os.environ["CLAUDE_CODE_HOST_SESSION_ID"] = ID
        _write_text(os.path.join(self.d, "watcher.alive"), "")
        _write_text(os.path.join(self.d, "mode"), "enforce\n")
        _write_text(os.path.join(self.d, "speed"), "inline\n")
        p = os.path.join(self.d, "s", "a", "b"); os.makedirs(p)
        _write_json(os.path.join(p, ID + ".json"), {"sessionId": ID, "cliSessionId": "cli1", "model": "claude-opus-5-5",
                                                    "title": "Some chat"})
        self.acpmh = os.path.join(self.d, "acpmh_root")
        os.makedirs(os.path.join(self.acpmh, "acpmh"))
        _write_text(os.path.join(self.acpmh, "acpmh", "__init__.py"), "")
        self.calls = os.path.join(self.d, "ocr_calls.jsonl")
        _write_text(os.path.join(self.acpmh, "acpmh", "images.py"),
                    "import json, os\n"
                    "def ocr(paths):\n"
                    "    with open(%r, 'a') as f: f.write(json.dumps(list(paths)) + '\\n')\n"
                    "    texts = json.loads(os.environ.get('FAKE_OCR', '{}'))\n"
                    "    return {p: texts[os.path.basename(p)] for p in paths if os.path.basename(p) in texts}\n" % self.calls)
        self.shots = []
        for name in ("1.png", "2.webp", "3.png", "4.png"):
            f = os.path.join(self.d, name); _write_text(f, "img"); self.shots.append(f)
        self.seen = []

    def tearDown(self):
        os.environ.pop("AUTOSWITCH_SESSIONS", None)

    def _scorer(self, label, prob):
        import http.server, threading
        seen = self.seen
        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(s):
                body = json.loads(s.rfile.read(int(s.headers["Content-Length"])))
                seen.append(body["text"]); s.send_response(200); s.end_headers()
                s.wfile.write(json.dumps({"label": label, "probs": {label: prob}}).encode())
            def log_message(s, *a): pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(_stop, srv)
        return f"http://127.0.0.1:{srv.server_port}"

    def _hook(self, prompt, images=None, ocr=None, label="opus", prob=0.9, **env):
        tr = os.path.join(self.d, "t.jsonl"); _write_text(tr, "x" * 400)
        hook = {"prompt": prompt, "session_id": "cli1", "transcript_path": tr}
        if images is not None:
            hook["images"] = images
        out = subprocess.run([sys.executable, os.path.join(SB, "hint.py")], capture_output=True, text=True,
                             env={**os.environ, "AUTOSWITCH_SCORER": self._scorer(label, prob), "AUTOSWITCH_DRY": "1",
                                  "AUTOSWITCH_ACPMH": self.acpmh, "FAKE_OCR": json.dumps(ocr or {}), **env},
                             input=json.dumps(hook))
        return json.loads(out.stdout) if out.stdout.strip() else None

    def _ocr_calls(self):
        return _read_jsonl(self.calls) if os.path.exists(self.calls) else []

    def test_image_only_prompt_scores_the_ocr_text(self):
        text = "Traceback: KeyError in the scheduler while rebalancing shards across regions"
        d = self._hook("", [self.shots[0]], {"1.png": text})
        self.assertIsNotNone(d)
        self.assertEqual((d["label"], d["screenshot"], d["ocr_chars"]), ("opus", True, len(text)))
        self.assertEqual(self.seen, ["\n\n[Text in the attached screenshot]\n" + text])

    def test_typed_text_and_image_are_scored_together(self):
        d = self._hook("why does this fail?", [self.shots[0]], {"1.png": "Error: connection refused on port 5432"})
        self.assertEqual(self.seen, ["why does this fail?\n\n[Text in the attached screenshot]\nError: connection refused on port 5432"])
        self.assertEqual(d["ocr_chars"], len("Error: connection refused on port 5432"))

    def test_image_without_readable_text_gets_no_decision(self):
        for ocr in ({}, {"1.png": "ok"}):
            d = self._hook("", [self.shots[0]], ocr, label="haiku", prob=0.99)
            self.assertIsNone(d["target"])
            self.assertEqual(d["reason"], "screenshot without readable text")
        self.assertEqual(self.seen, [])  # the empty string is never scored (it reads as Haiku 0.84)

    def test_empty_prompt_without_images_never_gets_a_target(self):
        for hook_images in (None, [], [os.path.join(self.d, "missing.png")]):
            self.assertIsNone(self._hook("", hook_images, label="haiku", prob=0.99))
        self.assertIsNone(self._hook("   ", label="haiku", prob=0.99))
        self.assertEqual(self.seen, [])

    def test_only_the_first_three_images_are_read(self):
        ocr = {os.path.basename(p): f"text of image {i} with enough words" for i, p in enumerate(self.shots)}
        d = self._hook("", self.shots, ocr)
        self.assertEqual(self._ocr_calls(), [self.shots[:3]])
        self.assertNotIn("image 3", self.seen[0])
        self.assertIn("image 0", self.seen[0]); self.assertIn("image 2", self.seen[0])
        self.assertTrue(d["screenshot"])

    def test_image_markers_in_the_prompt_are_read_and_not_scored(self):
        text = "Build failed: undefined symbol _vision_request in target Router"
        d = self._hook(f"[Image: source: {self.shots[1]}]", None, {"2.webp": text})
        self.assertEqual(self._ocr_calls(), [[self.shots[1]]])
        self.assertEqual(self.seen, ["\n\n[Text in the attached screenshot]\n" + text])
        self.assertEqual(d["ocr_chars"], len(text))

    def test_missing_acpmh_fails_soft(self):
        d = self._hook("explain the race in this lock code", [self.shots[0]], {"1.png": "x" * 50},
                       AUTOSWITCH_ACPMH=os.path.join(self.d, "nowhere"))
        self.assertEqual(self.seen, ["explain the race in this lock code"])  # scored on the typed text alone
        self.assertEqual(d["ocr_chars"], 0)
        d = self._hook("", [self.shots[0]], {"1.png": "x" * 50}, AUTOSWITCH_ACPMH=os.path.join(self.d, "nowhere"))
        self.assertEqual((d["target"], d["reason"]), (None, "screenshot without readable text"))

    def test_slow_ocr_is_cut_off(self):
        _write_text(os.path.join(self.acpmh, "acpmh", "images.py"),
                    "import time\ndef ocr(paths):\n    time.sleep(5)\n    return {p: 'late text that arrives too late' for p in paths}\n")
        t = time.time()
        d = self._hook("", [self.shots[0]], AUTOSWITCH_OCR_BUDGET="0.3")
        self.assertLess(time.time() - t, 3)
        self.assertEqual(d["reason"], "screenshot without readable text")

    def test_an_older_model_with_a_screenshot_moves_to_the_newest(self):
        _write_json(os.path.join(self.d, "s", "a", "b", ID + ".json"),
                    {"sessionId": ID, "cliSessionId": "cli1", "model": "claude-opus-4-7", "title": "Some chat"})
        d = self._hook("", [self.shots[0]], {"1.png": "design doc: storage layer tradeoffs and replication"}, prob=0.46)
        self.assertNotEqual(d["target"], "claude-haiku-5-5")
        self.assertIn(d["target"], ("claude-opus-5-5", None))


