#!/usr/bin/python

import sys
import time
import exceptions, traceback
import subprocess
import threading
import urllib, urllib2
import cookielib

http_proxy = 'http://10.10.44.251:6588'
https_proxy = 'http://10.10.44.251:6588'

class Browser(object):
    default_user_agent = "Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US; rv:1.9.1.8) Gecko/20100202 Firefox/3.5.8 GTB6 (.NET CLR 3.5.30729)"
    def __init__(self, user_agent=default_user_agent):
        self.headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5'
        }
        self.http_proxy = ""
        self.https_proxy = ""

    @staticmethod
    def make_url(base, **kw):
        return base + '?' + urllib.urlencode(kw)
        
    @staticmethod
    def make_opener(http_proxy, https_proxy):
        proxy_handler = urllib2.ProxyHandler({'http': http_proxy, 'https': https_proxy})
        proxy_auth_handler = urllib2.ProxyBasicAuthHandler()
        # proxy_auth_handler.add_password('realm', 'host', 'username', 'password')
        cookie=cookielib.CookieJar()
        return urllib2.build_opener(urllib2.HTTPCookieProcessor(cookie), proxy_handler, proxy_auth_handler)

    def get_page(self, url, data=None):
        opener = self.make_opener(self.http_proxy, self.https_proxy)
        if data: data = urllib.urlencode(data)
        request = urllib2.Request(url, data, self.headers)
        response = opener.open(request)
        return response.read()

class GoogleSearcher(Browser):
    def __init__(self):
        Browser.__init__(self)

    @staticmethod
    def make_search_url(query):
        return "http://www.google.com/search?hl=en&newwindow=1&%s&aq=f&aqi=g-e1g9&aql=&oq=" %(urllib.urlencode(dict(q=query)))
        
    def search(self, query):
        return self.get_page(self.make_search_url(query))
    
class Worker(threading.Thread):
    def __init__(self, interval, query):
        self.interval, self.query = interval, query
        threading.Thread.__init__(self)
        self.daemon = True
        
    def run(self):
        google_searcher = GoogleSearcher()
        google_searcher.http_proxy = http_proxy
        while True:
            try:
                f = open("result.html", 'w')
                f.write(google_searcher.search(self.query))
                f.close()
                subprocess.call("firefox result.html", shell=True)
            except exceptions.Exception,e:
                status = "error: %s" % e
                print status
                print traceback.format_exc(10)
                sys.exit()
            time.sleep(self.interval)


Worker(3, "what is love?").start()
while True:
    time.sleep(1)
sys.exit()
