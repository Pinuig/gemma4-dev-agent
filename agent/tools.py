"""Deterministic AST and repository tools. Test execution is NOT a security sandbox."""
from __future__ import annotations
import ast, fnmatch, os, subprocess, sys
from pathlib import Path
from typing import Any

IGNORED = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'node_modules', '.mypy_cache'}

def _root(repo_path: str) -> Path:
    root = Path(repo_path).expanduser().resolve()
    if not root.is_dir(): raise ValueError(f'Repository directory does not exist: {root}')
    return root

def _safe_path(repo_path: str, rel_path: str, *, must_exist: bool = False) -> Path:
    root = _root(repo_path)
    candidate = (root / rel_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError('Path escapes repository root')
    if must_exist and not candidate.is_file(): raise FileNotFoundError(rel_path)
    return candidate

def _iter_python(root: Path, file_glob: str = '*.py'):
    for path in root.rglob('*.py'):
        rel = path.relative_to(root)
        if any(part in IGNORED or part.startswith('.') for part in rel.parts): continue
        if fnmatch.fnmatch(path.name, file_glob): yield path

class _Definitions(ast.NodeVisitor):
    def __init__(self): self.items=[]; self.stack=[]
    def visit_ClassDef(self,node):
        self.items.append({'kind':'class','name':node.name,'qualified_name':'.'.join(self.stack+[node.name]),'lineno':node.lineno,'end_lineno':getattr(node,'end_lineno',node.lineno),'bases':[ast.unparse(x) for x in node.bases]})
        self.stack.append(node.name); self.generic_visit(node); self.stack.pop()
    def _function(self,node,kind):
        self.items.append({'kind':kind,'name':node.name,'qualified_name':'.'.join(self.stack+[node.name]),'lineno':node.lineno,'end_lineno':getattr(node,'end_lineno',node.lineno),'args':[a.arg for a in node.args.args]})
        self.stack.append(node.name); self.generic_visit(node); self.stack.pop()
    def visit_FunctionDef(self,node): self._function(node,'function')
    def visit_AsyncFunctionDef(self,node): self._function(node,'async_function')

class _Refs(ast.NodeVisitor):
    def __init__(self): self.names=set(); self.imports=set()
    def visit_Call(self,node):
        f=node.func
        if isinstance(f,ast.Name): self.names.add(f.id)
        elif isinstance(f,ast.Attribute): self.names.add(f.attr)
        self.generic_visit(node)
    def visit_Import(self,node): self.imports.update(a.name.split('.')[0] for a in node.names)
    def visit_ImportFrom(self,node):
        if node.module: self.imports.add(node.module.split('.')[0])

def search_symbol(repo_path: str, symbol_name: str, file_glob: str='*.py') -> list[dict[str,Any]]:
    root=_root(repo_path); results=[]
    for p in _iter_python(root,file_glob):
        try:
            src=p.read_text(encoding='utf-8'); tree=ast.parse(src); visitor=_Definitions(); visitor.visit(tree)
            for item in visitor.items:
                if symbol_name.lower() in item['name'].lower():
                    lines=src.splitlines(); a=max(0,item['lineno']-3); b=min(len(lines),item['end_lineno']+2)
                    item.update(file=str(p.relative_to(root)),snippet='\n'.join(f'{i+1}: {lines[i]}' for i in range(a,b)))
                    results.append(item)
        except (SyntaxError,UnicodeError): continue
    return results

def repository_map(repo_path: str, max_files: int=300) -> dict[str,Any]:
    root=_root(repo_path); files=[]; symbols=[]; edges=[]; imports={}
    for p in sorted(_iter_python(root)):
        if len(files)>=max_files: break
        rel=str(p.relative_to(root)); files.append(rel)
        try:
            src=p.read_text(encoding='utf-8'); tree=ast.parse(src); d=_Definitions(); r=_Refs(); d.visit(tree); r.visit(tree)
            for s in d.items: symbols.append({'file':rel,**s})
            imports[rel]=sorted(r.imports)
            for name in r.names: edges.append({'file':rel,'references':name})
        except (SyntaxError,UnicodeError): pass
    return {'root':str(root),'python_files':files,'symbols':symbols,'imports':imports,'references':edges}

def read_file(repo_path: str, rel_path: str, max_lines: int=500) -> str:
    p=_safe_path(repo_path,rel_path,must_exist=True); lines=p.read_text(encoding='utf-8',errors='replace').splitlines()
    if max_lines<1: raise ValueError('max_lines must be positive')
    return '\n'.join(f'{i+1:4}: {line}' for i,line in enumerate(lines[:max_lines])) + (f'\n...[{len(lines)-max_lines} lines omitted]' if len(lines)>max_lines else '')

def write_file(repo_path: str, rel_path: str, content: str) -> dict[str,Any]:
    if not isinstance(content,str): raise TypeError('content must be a string')
    p=_safe_path(repo_path,rel_path); p.parent.mkdir(parents=True,exist_ok=True)
    # Resolve again after creating parents to prevent writing through a symlink escape.
    p=_safe_path(repo_path,rel_path)
    if p.exists() and not p.is_file(): raise ValueError('Target is not a regular file')
    p.write_text(content,encoding='utf-8')
    return {'ok':True,'file':rel_path,'characters':len(content)}

def syntax_check(repo_path: str, rel_path: str) -> dict[str,Any]:
    p=_safe_path(repo_path,rel_path,must_exist=True)
    try: compile(p.read_text(encoding='utf-8'),str(p),'exec'); return {'ok':True,'message':'Syntax OK'}
    except SyntaxError as e: return {'ok':False,'message':f'{e.msg} at line {e.lineno}'}

def run_pytest(repo_path: str, test_path: str|None=None, keyword: str|None=None, timeout: int=60) -> dict[str,Any]:
    root=_root(repo_path)
    if test_path: _safe_path(repo_path,test_path,must_exist=True)
    if timeout<1 or timeout>300: raise ValueError('timeout must be between 1 and 300 seconds')
    cmd=[sys.executable,'-m','pytest','-q','--tb=short']
    if test_path: cmd.append(test_path)
    if keyword: cmd.extend(['-k',keyword])
    try:
        p=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=timeout,env={**os.environ,'PYTHONPATH':str(root)})
        out=(p.stdout+'\n'+p.stderr)[-6000:]
        return {'passed':p.returncode==0,'returncode':p.returncode,'output':out,'summary':'PASSED' if p.returncode==0 else 'FAILED'}
    except subprocess.TimeoutExpired as e: return {'passed':False,'returncode':-1,'output':str(e)[-2000:],'summary':'TIMEOUT'}

TOOLS={
 'repository_map':repository_map,'search_symbol':search_symbol,'read_file':read_file,
 'write_file':write_file,'syntax_check':syntax_check,'run_pytest':run_pytest,
}
TOOL_DESCRIPTIONS={
 'repository_map':'Build a deterministic Python AST repository map with files, definitions, imports and references.',
 'search_symbol':'Find function, async function or class definitions by partial name using Python AST.',
 'read_file':'Read a repository-relative file with line numbers.',
 'write_file':'Replace a repository-relative file with complete contents. Inspect it first.',
 'syntax_check':'Compile-check a Python file.',
 'run_pytest':'Run pytest with a timeout. Only use on trusted code; this is not a security sandbox.'}
def call_tool(name: str, repo_path: str, **kwargs):
    if name not in TOOLS: return {'error':f'Unknown tool: {name}'}
    try: return TOOLS[name](repo_path,**kwargs)
    except Exception as e: return {'error':f'{type(e).__name__}: {e}'}
