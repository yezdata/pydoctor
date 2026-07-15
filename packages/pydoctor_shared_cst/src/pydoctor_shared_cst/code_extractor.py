import libcst as cst
import re
from libcst.metadata import PositionProvider
from collections import deque
from typing import Literal


def is_property_or_special(node: cst.FunctionDef) -> bool:
    is_special = node.name.value.startswith("__") and node.name.value.endswith("__")
    is_property = any(
        isinstance(deco.decorator, cst.Name) and deco.decorator.value == "property"
        for deco in node.decorators
    )
    return is_special or is_property


def strip_docstring_from_node(
    node: cst.FunctionDef | cst.ClassDef,
) -> tuple[cst.FunctionDef | cst.ClassDef, str | None]:
    """Strip docstring from node and return the node and the docstring"""
    docstring = node.get_docstring()
    if docstring is None:
        return node, None

    body_elements = list(node.body.body)
    new_body = (
        body_elements[1:]
        if len(body_elements) > 1
        else [cst.SimpleStatementLine(body=[cst.Pass()])]
    )
    return node.with_changes(
        body=node.body.with_changes(body=new_body)
    ), docstring.strip()


class ClassSkeletonTransformer(cst.CSTTransformer):
    """Creates custom class code format"""

    def __init__(self):
        super().__init__()
        self.extracted_context = []
        self.pass_body = cst.IndentedBlock(
            body=[cst.SimpleStatementLine(body=[cst.Pass()])]
        )

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        new_body = []

        for item in updated_node.body.body:
            if (
                isinstance(item, cst.SimpleStatementLine)
                and len(item.body) == 1
                and isinstance(item.body[0], cst.Expr)
                and isinstance(item.body[0].value, cst.SimpleString)
            ):
                new_body.append(item)

            elif isinstance(item, cst.FunctionDef):
                if item.name.value == "__init__":
                    init_node, _ = strip_docstring_from_node(item)
                    new_body.append(init_node)
                else:
                    stub_method = item.with_changes(body=self.pass_body)
                    self.extracted_context.append(stub_method)

            elif isinstance(item, cst.SimpleStatementLine):
                new_body.append(item)

        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_body)
        )


class CodeExtractor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        extraction_options: Literal["with_docstring", "without_docstring", "all"],
    ):
        super().__init__()
        self.root_module = None

        self.extraction_options = extraction_options

        self._class_skeletons = {}
        self.stack = deque()

        # "line_col": {target, context}
        self.extracted_blocks = {}

    def visit_Module(self, node: cst.Module) -> None:
        self.root_module = node

    def _get_node_id(self, node: cst.CSTNode) -> str:
        """Generate unique ID based on code position"""
        pos = self.get_metadata(PositionProvider, node)
        return f"{pos.start.line}_{pos.start.column}"

    def _return_code(self, docstring: str | None) -> bool:
        if self.extraction_options == "with_docstring" and not docstring:
            return False
        elif self.extraction_options == "without_docstring" and docstring:
            return False
        return True

    def _has_ignore_comment(self, node: cst.FunctionDef | cst.ClassDef) -> bool:
        """Checks if the node has a '# pydoctor: ignore' comment either before or on the same line."""
        ignore_regex = re.compile(r"pydoctor:\s*ignore", re.IGNORECASE)

        if node.leading_lines:
            for line in node.leading_lines:
                if line.comment and ignore_regex.search(line.comment.value):
                    return True

        if hasattr(node, "body") and hasattr(node.body, "header"):
            trailing = node.body.header
            if trailing and hasattr(trailing, "comment") and trailing.comment:
                if ignore_regex.search(trailing.comment.value):
                    return True

        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if self._has_ignore_comment(node):
            return False

        node_id = self._get_node_id(node)

        full_name = node.name.value

        # is in class
        if len(self.stack) > 0 and isinstance(self.stack[-1], cst.ClassDef):
            if node.name.value == "__init__" or is_property_or_special(node):
                # self.stack.append(node)
                return False

            # not in __init__ -> method
            parent_class = self.stack[-1]
            full_name = f"{parent_class.name.value}.{node.name.value}"

            parent_id = self._get_node_id(parent_class)
            class_code = self._class_skeletons.get(parent_id, "")

            clean_node, docstring = strip_docstring_from_node(node)

            if not self._return_code(docstring):
                # self.stack.append(node)
                return False

            method_code = self.root_module.code_for_node(clean_node).strip()
            code = {
                "context": class_code,
                "target": method_code,
                "name": full_name,
            }

        else:
            clean_node, docstring = strip_docstring_from_node(node)
            if not self._return_code(docstring):
                # self.stack.append(node)
                return False

            func_code = self.root_module.code_for_node(clean_node).strip()
            code = {
                "context": "Independent code block",
                "target": func_code,
                "name": full_name,
            }

        self.extracted_blocks[node_id] = code
        # self.stack.append(node)
        return False

    # def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
    #     self.stack.pop()

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if self._has_ignore_comment(node):
            return False

        # dont extract nested classes
        if any(isinstance(n, cst.ClassDef) for n in self.stack):
            return False

        node_id = self._get_node_id(node)

        skeleton_transformer = ClassSkeletonTransformer()
        class_no_methods = node.visit(skeleton_transformer)

        clean_class_no_methods, docstring = strip_docstring_from_node(class_no_methods)

        self._class_skeletons[node_id] = self.root_module.code_for_node(
            clean_class_no_methods
        ).strip()

        self.stack.append(node)

        if not self._return_code(docstring):
            return True

        class_code = self.root_module.code_for_node(clean_class_no_methods).strip()
        method_context_code = self.root_module.code_for_node(
            cst.Module(body=skeleton_transformer.extracted_context)
        ).strip()

        self.extracted_blocks[node_id] = {
            "target": class_code,
            "context": method_context_code,
            "name": node.name.value,
        }

        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        if self.stack and self.stack[-1] is original_node:
            self.stack.pop()
