expr = '''
g_x = 1
def f():
  print g_x
'''
def exec_str(str):
    exec expr in globals(), globals()
dict = exec_str(expr)
print dict
print globals()
