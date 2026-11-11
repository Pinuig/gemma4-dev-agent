# 90-second Demo Video Script

**Goal:** Show one complete successful episode offline.

**Screen recording steps (terminal or VS Code + terminal):**

1. (0–5 s) Title card: “Gemma 4 DevAgent – Self-Correcting Coding Agent”
2. (5–15 s) Show the issue:
   ```
   Issue: The add function returns wrong results for negative numbers.
   Make test_add_negative pass.
   ```
3. (15–30 s) Agent calls `search_symbol("add")` → shows the wrong implementation (`return a - b`)
4. (30–45 s) Agent calls `write_file` with the corrected function (`return a + b`)
5. (45–55 s) `syntax_check` → OK
6. (55–70 s) `run_pytest -k test_add_negative` → FAILED (first attempt intentionally or show a small residual error)
7. (70–85 s) Agent reads the failure, issues a second corrected write, runs pytest again → PASSED
8. (85–90 s) Final frame: “SUCCESS – tests green” + GitHub / Kaggle links

**Recording tips:**
- Use a large font in the terminal
- Zoom on the tool calls and pytest output
- Add light background music or keep silent with captions
- Export at 1080p, ≤ 2 minutes

Upload to YouTube (unlisted or public) and paste the link into the Kaggle Writeup.
