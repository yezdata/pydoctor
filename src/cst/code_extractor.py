import ast
import libcst as cst
from libcst import matchers as m
from collections import deque

from src.utils.config_models import SpecialTokens


def is_property_or_special(node: cst.FunctionDef) -> bool:
    is_special = node.name.value.startswith("__") and node.name.value.endswith("__")
    is_property = any(
        m.matches(deco.decorator, m.Name("property")) for deco in node.decorators
    )
    return is_special or is_property


def get_code_docstring(
    module: cst.Module, node: cst.FunctionDef | cst.ClassDef
) -> tuple[str, str | None]:
    body_elements = list(node.body.body)

    if body_elements and m.matches(
        body_elements[0], m.SimpleStatementLine(body=[m.Expr(value=m.SimpleString())])
    ):
        docstring_text = ast.literal_eval(body_elements[0].body[0].value.value)

        new_body_elements = body_elements[1:]

        if not new_body_elements:
            new_body_elements = [cst.SimpleStatementLine(body=[cst.Pass()])]

        clean_node = node.with_changes(
            body=node.body.with_changes(body=new_body_elements)
        )
        return module.code_for_node(clean_node).strip(), docstring_text.strip()

    return module.code_for_node(node).strip(), None


class ClassSkeletonTransformer(cst.CSTTransformer):
    def __init__(self):
        super().__init__()
        self.add_methods: bool = True
        self.pass_body = cst.IndentedBlock(
            body=[cst.SimpleStatementLine(body=[cst.Pass()])]
        )

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        new_body = []

        for item in updated_node.body.body:
            if m.matches(
                item, m.SimpleStatementLine(body=[m.Expr(value=m.SimpleString())])
            ):
                new_body.append(item)

            elif isinstance(item, cst.FunctionDef):
                if item.name.value == "__init__":
                    new_body.append(item)
                else:
                    if self.add_methods:
                        stub_method = item.with_changes(body=self.pass_body)
                        new_body.append(stub_method)

            elif isinstance(item, cst.SimpleStatementLine):
                new_body.append(item)

        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_body)
        )


class CodeExtractor(cst.CSTVisitor):
    def __init__(self, module: cst.Module, spec_tokens: SpecialTokens, eos_token: int):
        self.module = module
        self.spec_tokens = spec_tokens
        self.eos_token = eos_token
        self.class_transformer = ClassSkeletonTransformer()
        self._class_skeletons = {}
        self.stack = deque()

        self.extracted_blocks = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        # is in class
        if len(self.stack) > 0 and isinstance(self.stack[-1], cst.ClassDef):
            if node.name.value == "__init__":
                self.stack.append(node)
                return False

            # don't extract
            if is_property_or_special(node):
                self.stack.append(node)
                return False

            # not in __init__ -> method
            parent_class = self.stack[-1]

            class_code = self._class_skeletons.get(id(parent_class), "")
            method_code, docstring = get_code_docstring(self.module, node)
            if not docstring:
                self.stack.append(node)
                return False

            code = f"{class_code}\n\n{method_code}"

            final_text = (
                f"{code}\n\n{self.spec_tokens.somd_token}{docstring}{self.eos_token}"
            )
        else:
            code, docstring = get_code_docstring(self.module, node)
            if not docstring:
                self.stack.append(node)
                return False

            final_text = (
                f"{code}\n\n{self.spec_tokens.sofd_token}{docstring}{self.eos_token}"
            )

        self.extracted_blocks.append({"text": final_text})
        self.stack.append(node)
        return False

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.stack.pop()

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self.class_transformer.add_methods = True
        modified_node = node.visit(self.class_transformer)

        code, docstring = get_code_docstring(cst.Module([]), modified_node)  # type: ignore
        if not docstring:
            self.stack.append(node)
            return False

        self.class_transformer.add_methods = False
        modified_class = node.visit(self.class_transformer)
        code_without_methods, _ = get_code_docstring(cst.Module([]), modified_class)  # type: ignore

        self._class_skeletons[id(node)] = code_without_methods

        final_text = (
            f"{code}\n\n{self.spec_tokens.socd_token}{docstring}{self.eos_token}"
        )

        self.extracted_blocks.append({"text": final_text})
        self.stack.append(node)
        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self.stack.pop()
