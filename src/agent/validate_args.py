"""原创练习 3：简化版工具参数校验器。

题目：实现 validate_args(args, schema)，返回全部错误组成的字符串列表。
合法时返回空列表；不使用 jsonschema。

本题只支持：
1. 顶层 object；properties 中的字段类型仅有 string/integer/number/boolean。
2. required 指定必填字段，默认没有必填字段。
3. additionalProperties=False 时禁止额外字段，默认允许额外字段。
4. 不考虑嵌套，保证 schema 本身合法；参数来自 JSON 解析，使用普通内置类型。
5. 必填字段存在但值为 None 时，仍要检查类型；True 不能算 integer/number。

错误顺序：先按 required 顺序报告缺失，再按 args 顺序检查多余字段和类型。
思路：先检查根对象与必填项，再逐个字段检查；用 type 精确区分 bool 和 int。
"""


def validate_args(args, schema):
    if not isinstance(args, dict):
        return ['参数必须是 object']

    types = {
        'string': (str,),
        'integer': (int,),
        'number': (int, float),
        'boolean': (bool,),
    }
    properties = schema.get('properties', {})
    errors = []

    for name in schema.get('required', []):
        if name not in args:
            errors.append(f'缺少必填字段：{name}')

    for name, value in args.items():
        if name not in properties:
            if schema.get('additionalProperties', True) is False:
                errors.append(f'不允许额外字段：{name}')
            continue
        expected = properties[name]['type']
        if type(value) not in types[expected]:
            errors.append(f'{name} 应为 {expected}，实际为 {type(value).__name__}')

    return errors


if __name__ == '__main__':
    schema = {
        'type': 'object',
        'properties': {
            'query': {'type': 'string'},
            'count': {'type': 'integer'},
            'verbose': {'type': 'boolean'},
        },
        'required': ['query', 'count'],
        'additionalProperties': False,
    }
    assert validate_args({'query': '北京', 'count': 3}, schema) == []
    answer = validate_args({'count': True, 'verbose': None, 'extra': 1}, schema)
    assert answer == [
        '缺少必填字段：query',
        'count 应为 integer，实际为 bool',
        'verbose 应为 boolean，实际为 NoneType',
        '不允许额外字段：extra',
    ]
    print(answer)
