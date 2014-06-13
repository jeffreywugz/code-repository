#!/usr/bin/python2

import sys
import time
import exceptions, traceback
import re
import subprocess
import threading
import urllib, urllib2
import cookielib
import HTMLParser
# fix the bug caused by "cjk encode"
HTMLParser.attrfind = re.compile(
      r'\s*([a-zA-Z_][-.:a-zA-Z_0-9]*)(\s*=\s*'
      r'(\'[^\']*\'|"[^"]*"|[-a-zA-Z0-9./,:;+*%?!&$\(\)_#=~@\xA1-\xFE]*))?')
from urlparse import urljoin

def write(path, content):
    with open(path, 'w') as f:
        f.write(content)
        
def popen(cmd):
    return Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT).communicate()[0]

class FormExtractor(HTMLParser.HTMLParser):
    def __init__(self, match_request):
        self.match_request, self.in_form, self.vars = match_request, False, {}
        HTMLParser.HTMLParser.__init__(self)

    def form_match(self, attrs):
        return set(self.match_request.items()) <= set(attrs)
        
    def handle_starttag(self, tag, attrs):
        if tag == 'form' and self.form_match(attrs):
            attrs = dict(attrs)
            self.action  = attrs["action"]
            self.in_form = True
        if not self.in_form: return
        if not (tag == 'input' and ('type', 'hidden') in attrs): return
        attrs = dict(attrs)
        self.vars[attrs['name']] = attrs['value']

    def handle_endtag(self, tag):
        if not self.in_form: return
        if tag == 'form': self.in_form = False

class Browser(object):
    default_user_agent = "Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US; rv:1.9.1.8) Gecko/20100202 Firefox/3.5.8 GTB6 (.NET CLR 3.5.30729)"
    def __init__(self, user_agent=default_user_agent, http_proxy=None, https_proxy=None):
        self.headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5'
        }
        self.http_proxy, self.https_proxy = http_proxy, https_proxy
        self.opener = self.make_opener(self.http_proxy, self.https_proxy)

    @staticmethod
    def make_url(base, **kw):
        return base + '?' + urllib.urlencode(kw)
        
    @staticmethod
    def make_opener(http_proxy, https_proxy):
        proxy = dict([(k,v) for k,v in ('http', http_proxy), ('https',https_proxy) if v])
        proxy_handler = urllib2.ProxyHandler(proxy)
        # proxy_auth_handler = urllib2.ProxyBasicAuthHandler()
        # proxy_auth_handler.add_password('realm', 'host', 'username', 'password')
        cj=cookielib.CookieJar()
        return urllib2.build_opener(proxy_handler, urllib2.HTTPCookieProcessor(cj))

    def get_page(self, url, data=None):
        if data: data = urllib.urlencode(data)
        request = urllib2.Request(url, data, self.headers)
        return self.opener.open(request).read()

    def form_submit(self, url, submit_vars, form_match_request={}):
        source_page = self.get_page(url)
        form_extractor = FormExtractor(form_match_request)
        form_extractor.feed(source_page)
        action = urljoin(url, form_extractor.action)
        form_extractor.vars.update(submit_vars)
        submit_vars = form_extractor.vars
        response = self.get_page(action, submit_vars)
        return self.maybe_jump(response)

    def maybe_jump(self, html):
        url_repr = re.search('location\.replace\("(.+)"\)', html)
        if not url_repr: return html
        url = eval("\'%s\'"%url_repr.group(1))
        return self.get_page(url)

def dinner_order(browser, name, passwd):
    url = 'http://bjdc.taobao.ali.com/'
    order_url = url + '/dingcan'
    f = browser.form_submit(url,  {'name':name, 'pass':passwd},
                            dict(id="user-login-form"))
    popen('msg %username% /time:7 "dinner order"')
    # write('result.html', browser.get_page(order_url))
    
    
if __name__ == '__main__':
    from Tkinter import *
    browser = Browser()
    dinner_order(browser, 'user', 'passwd')
