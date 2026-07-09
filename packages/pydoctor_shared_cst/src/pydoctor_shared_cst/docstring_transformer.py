import libcst as cst
from libcst.metadata import PositionProvider


from pydoctor_shared_cst.code_extractor import strip_docstring_from_node


class DocstringTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        generated_docstrings: dict[str, str],
    ):
        super().__init__()
        self.generated_docstrings = generated_docstrings
        self.transformed_blocks = 0

        self.old_docstrings = {}

    def _get_node_id(self, node: cst.CSTNode) -> str:
        """Generate unique ID based on code position"""
        pos = self.get_metadata(PositionProvider, node)
        return f"{pos.start.line}_{pos.start.column}"

    def insert_docstring(
        self, node: cst.FunctionDef | cst.ClassDef, docstring: str, orig_node_id: str
    ) -> cst.FunctionDef | cst.ClassDef:
        node, old_docstring = strip_docstring_from_node(node)

        self.old_docstrings[orig_node_id] = (
            old_docstring if old_docstring is not None else ""
        )

        docstring_node = cst.SimpleStatementLine(
            body=[cst.Expr(value=cst.SimpleString(value=docstring))]
        )
        new_body = [docstring_node] + list(node.body.body)
        return node.with_changes(body=node.body.with_changes(body=new_body))

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        orig_node_id = self._get_node_id(original_node)

        docstring = self.generated_docstrings.get(orig_node_id)
        if docstring is None:
            return updated_node

        self.transformed_blocks += 1
        return self.insert_docstring(updated_node, docstring, orig_node_id)

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        orig_node_id = self._get_node_id(original_node)

        docstring = self.generated_docstrings.get(orig_node_id)
        if docstring is None:
            return updated_node

        self.transformed_blocks += 1
        return self.insert_docstring(updated_node, docstring, orig_node_id)
