import libcst as cst
from libcst.metadata import PositionProvider


from pydoctor_shared_cst.code_extractor import strip_docstring_from_node


def insert_docstring(
    node: cst.FunctionDef | cst.ClassDef, docstring: str
) -> cst.CSTNode:
    node, _ = strip_docstring_from_node(node)

    docstring_node = cst.SimpleStatementLine(
        body=[cst.Expr(value=cst.SimpleString(value=docstring))]
    )
    new_body = [docstring_node] + list(node.body.body)
    return node.with_changes(body=node.body.with_changes(body=new_body))


class DocstringTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        generated_docstrings: dict[str, str],
    ):
        super().__init__()
        self.generated_docstrings = generated_docstrings

    def _get_node_id(self, node: cst.CSTNode) -> str:
        """Generate unique ID based on code position"""
        pos = self.get_metadata(PositionProvider, node)
        return f"{pos.start.line}_{pos.start.column}"

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        node_id = self._get_node_id(original_node)

        docstring = self.generated_docstrings.get(node_id)
        if docstring is None:
            return updated_node

        return insert_docstring(updated_node, docstring)

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        node_id = self._get_node_id(original_node)

        docstring = self.generated_docstrings.get(node_id)
        if docstring is None:
            return updated_node

        return insert_docstring(updated_node, docstring)
