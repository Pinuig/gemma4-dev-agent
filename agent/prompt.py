"""Prompt and robust JSON tool-call parsing."""
import json,re
TOOL_NAMES={'repository_map','search_symbol','read_file','write_file','syntax_check','run_pytest'}
SYSTEM_PROMPT='''You are Gemma 4 DevAgent, a careful Python software-engineering agent. Fix the reported issue in the supplied repository.
Available tools (call one at a time):
- repository_map(max_files=300): inspect Python files, AST definitions, imports and references.
- search_symbol(symbol_name, file_glob="*.py"): locate definitions.
- read_file(rel_path, max_lines=500): inspect source and tests.
- write_file(rel_path, content): replace a complete file. Inspect before editing; preserve unrelated behavior.
- syntax_check(rel_path): validate Python syntax.
- run_pytest(test_path=null, keyword=null, timeout=60): run tests.
Workflow: map repository; inspect relevant tests and implementation; form a minimal fix; write; syntax-check; run focused tests and then full tests when practical. On failure, use actual traceback/output to revise. Never claim success unless pytest passed after the latest change. Do not modify tests merely to hide a defect. Do not invent observations. Work only inside the repository. Tool calls must use this exact format:
<tool_call>{{"name":"TOOL_NAME","arguments":{{...}}}}</tool_call>
When tests pass, respond: SUCCESS: <concise change summary and tests run>.
If unable to fix, explain the blocker and evidence.
ISSUE: {issue}
REPOSITORY: {repo_path}
'''
def build_system_prompt(issue,repo_path): return SYSTEM_PROMPT.format(issue=issue,repo_path=repo_path)
def format_tool_call(name,arguments): return '<tool_call>'+json.dumps({'name':name,'arguments':arguments},ensure_ascii=False)+'</tool_call>'
def parse_tool_call(text):
    match=re.search(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S)
    if not match:return None,{}
    raw=match.group(1).strip()
    try:
        obj=json.loads(raw)
        if isinstance(obj,dict) and 'name' in obj:
            name=obj['name']; args=obj.get('arguments',{})
        else: return None,{}
        if name not in TOOL_NAMES or not isinstance(args,dict): return None,{}
        return name,args
    except (ValueError,TypeError): return None,{}
