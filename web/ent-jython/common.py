import sys,os, os.path
import exceptions
import subprocess
import threading
import re
import copy

class GErr(exceptions.Exception):
    def __init__(self, msg, obj=None):
        exceptions.Exception(self)
        self.obj, self.msg = obj, msg

    def __str__(self):
        return "%s\n%s"%(self.msg, self.obj)

def getObjDict(obj):
    def safe_getattr(obj, x):
        try:
            return getattr(obj,x)
        except exceptions.AttributeError:
            return None
    return dict([(x, safe_getattr(obj, x)) for x in dir(obj)])
    
def my_help(obj):
    objDict = getObjDict(obj)
    attrs = filter(callable, objDict.values())

    methods = filter(lambda x: not callable(x), objDict.values())
    print obj
    print '---Attrs---'
    for i in attrs:
        if i.startswith('__'): continue
        print i
    print '---Mehotds---'
    for i in methods:
        if i.startswith('__'): continue
        print i + '()'

class ObjFormatter:
    def __init__(self, obj):
        self.obj = obj

    def getattr(self, attr):
        return getattr(self, attr, getattr(self.obj, attr, None))

    def slot_repr(self, attr):
        return '%s = %s'%(attr, repr(self.getattr(attr)))
    
    @staticmethod
    def indent(str):
        return '    ' + str.replace('\n', '\n    ')

    def format(self, constructor_args, obj_attrs):
        constructor = "%s(%s):\n"%(self.obj.__class__.__name__, ', '.join([self.slot_repr(i) for i in constructor_args]))
        attrs = '\n'.join([self.indent(self.slot_repr(i)) for i in obj_attrs])
        return constructor + attrs
        
def short_repr(x):
    if len(x) < 80: return x
    else: return x[:80] + '...'
    
def traceit(func):
    def wrapper(*args, **kw):
        args_repr = [repr(arg) for arg in args]
        kw_repr = ['%s=%s'%(k, repr(v)) for k,v in kw]
        full_repr = map(short_repr, args_repr + kw_repr)
        print '%s(%s)'%(func.__name__, ', '.join(full_repr))
        result = func(*args, **kw)
        print '=> %s'%(repr(result))
        return result
    return wrapper

def makeDynamicObj(objtype, **attrs):
    obj = objtype()
    [setattr(obj, key, value) for (key, value) in attrs.items()]
    return obj

def idSequence():
    lock = threading.Lock()
    counter = 0
    while True:
        lock.acquire()
        counter += 1
        yield counter
        lock.release()

def parse_cmd_args(args):
    splited_args = [i.split('=', 1) for i in args]
    list_args = [i[0] for i in splited_args if len(i)==1]
    kw_args = dict([i for i in splited_args if len(i)==2])
    return list_args, kw_args

def make_cmd_args(*list_args, **kw_args):
    args_repr = [repr(arg) for arg in list_args]
    kw_repr = ['%s=%s'%(k, repr(v)) for k,v in kw]
    return args_repr + kw_repr

def cmd_eval(env, func, *list_args, **kw_args):
    try:
        func = eval(func, env)
    except exceptions.IndexError:
        raise GErr('run_cmd(): need to specify a callable object.', args)
    if not callable(func):
        if list_args or kw_args: raise GErr("not callable", func)
        else: return func
    return func(*list_args, **kw_args)

def shell(cmd):
    ret = subprocess.call(cmd, shell=True)
    sys.stdout.flush()
    return ret

def popen(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    err = p.stderr.read()
    out = p.stdout.read()
    if p.returncode != 0 or err:
        raise GErr('%s\n%s'%(err, out), cmd)
    return out

def cmd_app_run(env, func, args, init=''):
    list_args, kw_args = parse_cmd_args(args)
    if kw_args.has_key('--init'):
        init = kw_args['--init']
        del kw_args['--init']
    exec init in env
    new_env = copy.copy(env)
    new_env.update(locals())
    result = cmd_eval(new_env, func, *list_args, **kw_args)
    return result

def cmd_app_call(func, init='', popen=popen, *list_args, **kw_args):
    if init: kw_args.update({'--init': init})
    args = make_cmd_args(*list_args, **kw_args)
    cmd = '%s %s'%(func, ' '.join(args))
    content = popen(cmd)
    return content

class MarkCodec:
    start_tag = '------------------------------/%s/------------------------------'
    end_tag = '------------------------------\%s\------------------------------'
    def __init__(self, tag='mark'):
        self.tag = tag

    def dumps(self, value):
        return '%s\n%s\n%s' %(self.start_tag % self.tag, repr(value), self.end_tag % self.tag)

    def loads(self, content, default=None):
        match = re.search(r'%s\n(.*)\n%s'%(self.start_tag % self.tag, self.end_tag % self.tag), content, re.S)
        if not match:
            return default
        expr = match.group(1)
        return eval(expr)

class CmdObjProxy:
    def __init__(self, base, init, tag='mark'):
        self.base, self.init, self.tag = base, init, tag

    def call(self, func, *arg, **kw):
        content = cmd_app_call('.'.join([self.base, func]), init=self.init, *args, **kw)
        mark = MarkCodec(self.tag)
        return mark.loads(content)

    def asyncall(self, func, log, *arg, **kw):
        def bg_popen(cmd):
            return popen('%s >%s 2>&1'%(cmd, log))
        return cmd_app_call('.'.join([self.base, func]), popen=bg_popen, init=self.init, *args, **kw)
        
    def __getattr__(self, name):
        def wrapper(*args, **kw):
            return self.call(name, *args, **kw)
        return wrapper
