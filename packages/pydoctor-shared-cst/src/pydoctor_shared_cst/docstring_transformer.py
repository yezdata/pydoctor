import libcst as cst


class DocstringTransformer(cst.CSTTransformer):
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if updated_node.get_docstring() is not None:
            return updated_node

        string_node = cst.SimpleString('"""Nový robustní docstring."""')

        expr_node = cst.Expr(value=string_node)

        docstring_statement = cst.SimpleStatementLine(
            body=[expr_node],
            trailing_whitespace=cst.TrailingWhitespace(newline=cst.Newline()),
        )

        new_body = [docstring_statement] + list(updated_node.body.body)

        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_body)
        )
