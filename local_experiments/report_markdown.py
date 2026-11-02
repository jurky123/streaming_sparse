"""Convert the local research report to Markdown without external dependencies."""
from html.parser import HTMLParser
from pathlib import Path
import re


class Node:
    def __init__(self, tag, attrs=()):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children = []


class ReportParser(HTMLParser):
    VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node('root')
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def html_to_markdown(source, directory):
    parser = ReportParser()
    parser.feed(source)
    directory = Path(directory).resolve()
    anchors = {'results': '实测结果', 'videos': '逐场景视频',
               'survey': '方法调研与选择依据', 'protocol': '实验口径与原始实现差异',
               'reproduce': '复现与目录'}

    def target(value):
        if value.startswith('#'):
            return '#' + anchors.get(value[1:], value[1:])
        if value.startswith(('https://', 'http://')):
            return value
        value = str((directory / value).resolve())
        return '<' + value + '>' if ' ' in value else value

    def raw(node):
        return node if isinstance(node, str) else ''.join(raw(c) for c in node.children)

    def table_rows(node):
        if isinstance(node, str):
            return []
        if node.tag == 'tr':
            return [[re.sub(r'\s+', ' ', render(c)).strip().replace('|', '\\|')
                     for c in node.children if isinstance(c, Node) and c.tag in ('th', 'td')]]
        return [row for child in node.children for row in table_rows(child)]

    def table(rows, columns=None):
        columns = columns if columns is not None else range(len(rows[0]))
        lines = ['| ' + ' | '.join(row[i] for i in columns) + ' |' for row in rows]
        lines.insert(1, '| ' + ' | '.join('---' for _ in columns) + ' |')
        return '\n\n' + '\n'.join(lines) + '\n\n'

    def render(node):
        if isinstance(node, str):
            return re.sub(r'\s+', ' ', node)
        tag = node.tag
        if tag in ('style', 'script', 'title', 'meta'):
            return ''
        if tag == 'pre':
            content = raw(node).strip()
            language = 'bash' if content.startswith('cd /') else 'text'
            return '\n\n```' + language + '\n' + content + '\n```\n\n'
        if tag == 'table':
            rows = table_rows(node)
            if len(rows[0]) == 11 and rows[0][0] == '方法':
                return ('\n\n### 资源与耗时\n' + table(rows, [0, 1, 2, 3, 4, 10])
                        + '\n### 画面描述指标\n' + table(rows, [0, 5, 6, 7, 8, 9]))
            return table(rows)
        content = ''.join(render(c) for c in node.children)
        if tag in ('h1', 'h2', 'h3', 'h4'):
            return '\n\n' + '#' * int(tag[1]) + ' ' + content.strip() + '\n\n'
        if tag == 'p':
            return '\n\n' + content.strip() + '\n\n'
        if tag in ('strong', 'b'):
            return '**' + content.strip() + '**'
        if tag in ('em', 'i'):
            return '*' + content.strip() + '*'
        if tag == 'code':
            return '`' + raw(node).strip() + '`'
        if tag == 'a':
            return '[' + content.strip() + '](' + target(node.attrs['href']) + ')'
        if tag == 'img':
            return '\n\n![' + node.attrs.get('alt', '图表') + '](' + target(node.attrs['src']) + ')\n\n'
        if tag == 'video':
            return '\n\n[查看四宫格对比视频](' + target(node.attrs['src']) + ')\n\n'
        if tag == 'summary':
            return '\n\n**' + content.strip() + '**\n\n'
        if tag == 'li':
            return '\n- ' + content.strip() + '\n'
        if tag in ('ul', 'ol', 'details', 'div'):
            return '\n\n' + content.strip() + '\n\n'
        if tag == 'br':
            return ' · '
        return content

    result = render(parser.root)
    result = re.sub(r'\n[ \t]+', '\n', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip() + '\n'
