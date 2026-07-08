uv run --no-project \
  --with "Nuitka[onefile]" \
  --with "libcst" \
  --with "llama-cpp-python" \
  python -m nuitka --standalone --onefile --include-package=libcst --include-package=llama_cpp --report=report.xml packages/pydoctor_cli/src/pydoctor_cli/cli.py