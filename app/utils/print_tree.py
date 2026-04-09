# scripts/print_tree.py
from pathlib import Path

def show_tree(root: Path, prefix=""):
    for path in sorted(root.iterdir()):
        if path.name in {".git", "__pycache__", ".venv", "node_modules"}:
            continue

        print(prefix + ("📁 " if path.is_dir() else "📄 ") + path.name)

        if path.is_dir():
            show_tree(path, prefix + "   ")

if __name__ == "__main__":
    show_tree(Path(r"D:\Working\Cygnus\compass-voice"))
