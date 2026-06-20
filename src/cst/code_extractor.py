import ast
import libcst as cst
from libcst import matchers as m
from collections import deque


def get_code_docstring(
    module: cst.Module, node: cst.FunctionDef
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
    def __init__(self, module: cst.Module):
        self.module = module
        self.class_transformer = ClassSkeletonTransformer()
        self._class_skeletons = {}
        self.stack = deque()

        self.extracted_pairs = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        # is in class, but not __init__ -> method
        if len(self.stack) > 0 and isinstance(self.stack[-1], cst.ClassDef):
            if node.name.value == "__init__":
                self.stack.append(node)
                return False

            parent_class = self.stack[-1]

            class_code = self._class_skeletons.get(id(parent_class), "")
            method_code, docstring = get_code_docstring(self.module, node)
            if not docstring:
                self.stack.append(node)
                return False

            code = f"{class_code}{method_code}"
        else:
            code, docstring = get_code_docstring(self.module, node)
            if not docstring:
                self.stack.append(node)
                return False

        self.extracted_pairs.append({"code": code, "label": docstring})
        self.stack.append(node)
        return False

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.stack.pop()

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self.class_transformer.add_methods = True
        modified_node = node.visit(self.class_transformer)

        code, docstring = get_code_docstring(cst.Module([]), modified_node)
        if not docstring:
            self.stack.append(node)
            return False

        self.class_transformer.add_methods = False
        modified_class = node.visit(self.class_transformer)
        code_without_methods, _ = get_code_docstring(cst.Module([]), modified_class)

        self._class_skeletons[id(node)] = code_without_methods

        self.extracted_pairs.append({"code": code, "label": docstring})
        self.stack.append(node)
        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self.stack.pop()
