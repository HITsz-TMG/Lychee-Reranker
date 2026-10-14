import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_release.py"
SPEC = importlib.util.spec_from_file_location("lychee_check_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECK_RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_RELEASE)


class ReleaseScanTests(unittest.TestCase):
    def _git(self, root: Path, *arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory(prefix="release-scan-")
        root = Path(temporary.name)
        self._git(root, "init", "-q")
        self._git(root, "config", "user.name", "release scanner")
        self._git(root, "config", "user.email", "scanner@example.invalid")
        return temporary, root

    def test_checks_staged_blob_and_worktree_separately_without_printing_values(self):
        temporary, root = self._repo()
        with temporary:
            tracked = root / "worktree.txt"
            tracked.write_text("safe\n", encoding="utf-8")
            self._git(root, "add", "worktree.txt")
            private_path = "/" + "home" + "/alice/private"
            tracked.write_text(private_path + "\n", encoding="utf-8")

            staged = root / "index.txt"
            token = "sk-" + "q" * 24
            staged.write_text("api" + "_" + "key = \"" + token + "\"\n", encoding="utf-8")
            self._git(root, "add", "index.txt")
            staged.write_text("safe\n", encoding="utf-8")

            errors, checked = CHECK_RELEASE.scan_repository(root)
            joined = "\n".join(errors)
            self.assertGreaterEqual(checked, 4)
            self.assertIn("worktree:worktree.txt: private absolute path", joined)
            self.assertIn("index:index.txt: credential assignment", joined)
            self.assertNotIn(token, joined)
            self.assertNotIn(private_path, joined)

    def test_history_finds_deleted_main_file_and_symlink_but_ignores_side_branch(self):
        temporary, root = self._repo()
        with temporary:
            (root / "README.txt").write_text("safe\n", encoding="utf-8")
            self._git(root, "add", "README.txt")
            self._git(root, "commit", "-qm", "base")
            self._git(root, "branch", "-M", "main")

            self._git(root, "checkout", "-qb", "private-worker")
            side_path = "/" + "home" + "/worker/side"
            (root / "side.txt").write_text(side_path, encoding="utf-8")
            self._git(root, "add", "side.txt")
            self._git(root, "commit", "-qm", "private worker")

            self._git(root, "checkout", "-q", "main")
            historical_path = "/" + "root" + "/release/old"
            (root / "deleted.txt").write_text(historical_path, encoding="utf-8")
            self._git(root, "add", "deleted.txt")
            link = root / "historical-link"
            link_target = "/" + "path" + "/to/fixture"
            link.symlink_to(link_target)
            self._git(root, "add", "historical-link")
            self._git(root, "commit", "-qm", "release candidate")
            self._git(root, "rm", "-q", "deleted.txt", "historical-link")
            self._git(root, "commit", "-qm", "remove private files")

            errors, _ = CHECK_RELEASE.scan_repository(root, history=True)
            joined = "\n".join(errors)
            self.assertIn("deleted.txt: private absolute path", joined)
            self.assertIn("historical-link: symlink", joined)
            self.assertNotIn("side.txt", joined)
            self.assertNotIn(historical_path, joined)
            self.assertNotIn(side_path, joined)

    def test_symlink_artifact_and_credential_patterns_are_fail_closed(self):
        temporary, root = self._repo()
        with temporary:
            dangling = root / "dangling-link"
            dangling.symlink_to("/" + "tmp" + "/missing-fixture")
            (root / "weights.safetensors").write_bytes(b"fixture weights")
            secret = "AKIA" + "A" * 16
            private_key_marker = "BEGIN " + "PRIVATE KEY"
            aws_assignment = "AWS_" + "SECRET" + "_ACCESS_KEY=" + "z" * 24
            (root / "credentials.txt").write_text(
                private_key_marker + "\n" + aws_assignment + "\n" + "access" + "_" + "token=" + secret,
                encoding="utf-8",
            )
            (root / "safe-fixtures.txt").write_text(
                "/lib /usr/lib /path/to/model 127.0.0.1 localhost ::1\n",
                encoding="utf-8",
            )

            errors, _ = CHECK_RELEASE.scan_repository(root)
            joined = "\n".join(errors)
            self.assertIn("dangling-link: symlink", joined)
            self.assertIn("weights.safetensors: generated artifact", joined)
            self.assertIn("credentials.txt: private key", joined)
            self.assertIn("credentials.txt: cloud credential", joined)
            self.assertIn("credentials.txt: credential assignment", joined)
            self.assertNotIn(secret, joined)
            self.assertNotIn("safe-fixtures.txt", joined)


if __name__ == "__main__":
    unittest.main()
